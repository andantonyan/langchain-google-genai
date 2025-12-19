import { FunctionCall, GenerateContentResponse, UsageMetadata as GoogleUsageMetadata, Part } from '@google/genai';
import { AIMessage, AIMessageChunk, type MessageContent, UsageMetadata } from '@langchain/core/messages';
import { ToolCallChunk } from '@langchain/core/messages/tool';
import { ChatGeneration, ChatGenerationChunk } from '@langchain/core/outputs';
import { v4 as uuidv4 } from 'uuid';

/**
 * Extracts usage metadata from Google's format to LangChain's format.
 */
function extractUsageMetadata(usage?: GoogleUsageMetadata): UsageMetadata | undefined {
  if (!usage) return undefined;
  return {
    input_tokens: usage.promptTokenCount ?? 0,
    output_tokens: usage.responseTokenCount ?? 0,
    total_tokens: usage.totalTokenCount ?? 0,
  };
}

/**
 * Converts a Google FunctionCall to a LangChain ToolCallChunk.
 */
function convertFunctionCallToToolCallChunk(fc: FunctionCall, index: number): ToolCallChunk {
  return {
    name: fc.name ?? '',
    args: JSON.stringify(fc.args),
    id: fc.id ?? uuidv4(),
    index,
    type: 'tool_call_chunk',
  };
}

/**
 * Converts a single Google Part to an AIMessageChunk.
 */
function convertPartToChunk(part: Part, index: number): AIMessageChunk {
  const responseMetadata: Record<string, unknown> = {};

  if (part.thoughtSignature) {
    responseMetadata['thoughtSignature'] = part.thoughtSignature;
  }

  if (part.text !== undefined) {
    // If 'thought' is true, return it as a reasoning content block
    if (part.thought) {
      // We assign index 0 to reasoning blocks to ensure they merge correctly
      // during streaming concatenation.
      return new AIMessageChunk({
        content: [{ type: 'reasoning', reasoning: part.text, index: 0 }],
        response_metadata: responseMetadata,
      });
    }

    // We assign index 1 to text blocks and wrap them in an array.
    // This ensures that when a text chunk follows a reasoning chunk (which is an array),
    // the text chunk is merged into the content array rather than causing type conflicts
    // or fragmentation during concatenation.
    return new AIMessageChunk({
      content: part.text,
      response_metadata: responseMetadata,
    });
  }

  if (part.functionCall !== undefined) {
    return new AIMessageChunk({
      content: '',
      tool_call_chunks: [convertFunctionCallToToolCallChunk(part.functionCall, index)],
      response_metadata: responseMetadata,
    });
  }

  // If we only have metadata (e.g. just a signature update)
  if (Object.keys(responseMetadata).length > 0) {
    return new AIMessageChunk({
      content: '',
      response_metadata: responseMetadata,
    });
  }

  return new AIMessageChunk({ content: '' });
}

/**
 * Processes a stream chunk from Google GenAI and converts it to a ChatGenerationChunk.
 */
export function convertGoogleStreamChunkToLangChainChunk(
  response: GenerateContentResponse,
): ChatGenerationChunk | null {
  const candidate = response.candidates?.[0];
  if (!candidate) {
    // It might be a pure usage metadata chunk at the end
    if (response.usageMetadata) {
      return new ChatGenerationChunk({
        message: new AIMessageChunk({
          content: '',
          usage_metadata: extractUsageMetadata(response.usageMetadata),
        }),
        text: '',
      });
    }
    return null;
  }

  const chunk = candidate.content?.parts?.reduce((acc: AIMessageChunk | null, part, index) => {
    const nextChunk = convertPartToChunk(part, index);
    return acc ? acc.concat(nextChunk) : nextChunk;
  }, null);

  if (!chunk) return null;

  // Attach usage metadata if present in this chunk
  if (response.usageMetadata) {
    chunk.usage_metadata = extractUsageMetadata(response.usageMetadata);
  }

  // Determine text content for the chunk.
  // If the chunk is purely reasoning, text should be empty to avoid polluting the main output.
  // If the chunk is standard text, it should be populated.
  let chunkText = '';
  if (typeof chunk.content === 'string') {
    chunkText = chunk.content;
  } else if (Array.isArray(chunk.content)) {
    // Filter for text blocks only, ignoring reasoning blocks for the 'text' field
    chunkText = chunk.content
      .filter(block => block.type === 'text' && 'text' in block)
      .map(block => block.text)
      .join('');
  }

  return new ChatGenerationChunk({
    message: chunk,
    text: chunkText,
  });
}

/**
 * Converts a full non-streaming Google GenAI response to a ChatGeneration.
 */
export function convertGoogleResponseToChatGeneration(response: GenerateContentResponse): ChatGeneration {
  const candidate = response.candidates?.[0];
  if (!candidate || !candidate.content) {
    throw new Error('No candidates returned from Google GenAI.');
  }

  let textContent = '';
  const contentBlocks: MessageContent = [];
  const toolCalls = [];
  const responseMetadata: Record<string, unknown> = {
    finishReason: candidate.finishReason,
    index: candidate.index,
    ...response.promptFeedback, // Include safety feedback
  };

  if (candidate.content.parts) {
    for (const part of candidate.content.parts) {
      if (part.text) {
        if (part.thought) {
          // It's a reasoning block
          contentBlocks.push({ type: 'reasoning', reasoning: part.text });

          // Also accumulate in metadata for legacy access/convenience
          const existingThoughts = (responseMetadata['thoughts'] as string[]) || [];
          existingThoughts.push(part.text || '');
          responseMetadata['thoughts'] = existingThoughts;
        } else {
          // It's standard text
          textContent += part.text;
          contentBlocks.push({ type: 'text', text: part.text });
        }
      }

      if (part.functionCall) {
        toolCalls.push({
          name: part.functionCall.name ?? '',
          args: part.functionCall.args ?? {},
          id: part.functionCall.id ?? uuidv4(),
          type: 'tool_call' as const,
        });
      }

      // Critical: Capture Thought Signature
      if (part.thoughtSignature) {
        responseMetadata['thoughtSignature'] = part.thoughtSignature;
      }
      // Handle Code Execution
      if (part.executableCode) {
        responseMetadata['executableCode'] = part.executableCode;
      }
      if (part.codeExecutionResult) {
        responseMetadata['codeExecutionResult'] = part.codeExecutionResult;
      }
    }
  }

  // If we have reasoning blocks, return content as an array of blocks.
  // Otherwise, return simple string for better compatibility with standard chains.
  const hasReasoning = contentBlocks.some(b => b.type === 'reasoning');
  const finalContent = hasReasoning ? contentBlocks : textContent;

  const msg = new AIMessage({
    content: finalContent,
    tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
    response_metadata: responseMetadata,
    usage_metadata: extractUsageMetadata(response.usageMetadata),
  });

  return {
    text: textContent,
    message: msg,
    generationInfo: {
      finishReason: candidate.finishReason,
      index: candidate.index,
    },
  };
}
