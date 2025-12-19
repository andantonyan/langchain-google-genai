import { type Interactions } from '@google/genai';
import {
  AIMessage,
  AIMessageChunk,
  BaseMessage,
  ChatMessage,
  HumanMessage,
  MessageContent,
  SystemMessage,
  ToolMessage,
  UsageMetadata,
} from '@langchain/core/messages';
import { ToolCall } from '@langchain/core/messages/tool';
import { ChatGeneration } from '@langchain/core/outputs';

/**
 * Formats LangChain content (string or array) into Google Content Parts.
 */
function formatContent(content: MessageContent): Array<Interactions.TextContent | Interactions.ImageContent> {
  if (typeof content === 'string') {
    return [{ type: 'text', text: content }];
  }

  if (Array.isArray(content)) {
    const parts = content.map(part => {
      const type = part['type'];
      if (type === 'text') {
        const textPart: Interactions.TextContent = {
          type: 'text',
          text: (part['text'] as string) || '',
        };
        return textPart;
      }
      if (type === 'image_url') {
        let data: string | undefined;
        let mimeType: string | undefined;
        let uri: string | undefined;
        let url: string;

        const imageUrl = part['image_url'];
        if (typeof imageUrl === 'string') {
          url = imageUrl;
        } else if (imageUrl && typeof imageUrl === 'object' && 'url' in imageUrl) {
          url = imageUrl['url'] as string;
        } else {
          throw new Error('Invalid image_url format');
        }

        if (url.startsWith('data:')) {
          const matches = url.match(/^data:(.+);base64,(.+)$/);
          if (matches) {
            mimeType = matches[1];
            data = matches[2];
          } else {
            throw new Error('Invalid data URL format');
          }
        } else {
          uri = url;
        }

        const imageContent: Interactions.ImageContent = {
          type: 'image',
        };

        if (data && mimeType) {
          imageContent.data = data;
          imageContent.mime_type = mimeType as Interactions.ImageMimeType;
        }
        if (uri) {
          imageContent.uri = uri;
        }
        return imageContent;
      }
      return undefined;
    });

    // Filter undefined and assert type
    return parts.filter((part): part is Interactions.TextContent | Interactions.ImageContent => part !== undefined);
  }

  return [];
}

/**
 * Formats the result of a tool call.
 * Google expects `result` to be an object, string, or list of items.
 */
function formatToolResult(content: MessageContent): unknown {
  if (typeof content === 'string') {
    try {
      return JSON.parse(content);
    } catch {
      return content;
    }
  }
  return content;
}

/**
 * Converts a LangChain BaseMessage to a Google Interactions API Turn.
 * Handles mapping of roles, content parts, tool calls, and thought signatures.
 */
function convertMessageToGoogleTurn(message: BaseMessage): Interactions.Turn | null {
  if (HumanMessage.isInstance(message)) {
    return {
      role: 'user',
      content: formatContent(message.content),
    };
  } else if (AIMessage.isInstance(message)) {
    const parts: Array<
      | Interactions.TextContent
      | Interactions.ImageContent
      | Interactions.FunctionCallContent
      | Interactions.ThoughtContent
    > = [];

    // Handle Thought Signature (Critical for Gemini 3)
    const thoughtSignature = message.response_metadata?.['thought_signature'];
    if (typeof thoughtSignature === 'string') {
      parts.push({
        type: 'thought' as const,
        signature: thoughtSignature,
      });
    }

    // Handle Text Content
    if (typeof message.content === 'string' && message.content !== '') {
      parts.push({
        type: 'text' as const,
        text: message.content,
      });
    } else if (Array.isArray(message.content)) {
      parts.push(...formatContent(message.content));
    }

    // Handle Tool Calls
    const toolCalls = message.tool_calls;
    if (toolCalls && toolCalls.length > 0) {
      for (const toolCall of toolCalls) {
        parts.push({
          type: 'function_call' as const,
          id: toolCall.id ?? '',
          name: toolCall.name,
          arguments: toolCall.args,
        });
      }
    }

    return {
      role: 'model',
      content: parts,
    };
  } else if (ToolMessage.isInstance(message)) {
    return {
      role: 'user', // Tool results are sent as 'user' role in Interactions API
      content: [
        {
          type: 'function_result' as const,
          call_id: message.tool_call_id,
          name: message.name,
          result: formatToolResult(message.content),
          is_error: message.status === 'error',
        },
      ],
    };
  } else if (SystemMessage.isInstance(message)) {
    // System messages are handled separately in the top-level request
    return null;
  } else if (ChatMessage.isInstance(message)) {
    return {
      role: message.role === 'assistant' ? 'model' : (message.role as string),
      content: formatContent(message.content),
    };
  }

  throw new Error(`Unsupported message type: ${message.type}`);
}

/**
 * Converts a list of LangChain messages to the Google Interactions API payload.
 * Separates system instructions from the conversation history.
 */
export function convertMessagesToGoogleInteractionPayload(messages: BaseMessage[]): {
  input: Interactions.Turn[];
  system_instruction?: string;
} {
  const input: Interactions.Turn[] = [];
  let system_instruction: string | undefined;

  for (const message of messages) {
    if (SystemMessage.isInstance(message)) {
      let content = '';
      if (typeof message.content === 'string') {
        content = message.content;
      } else if (Array.isArray(message.content)) {
        content = message.content
          .filter(block => block.type === 'text')
          .map(block => (block as { text: string }).text)
          .join('\n');
      }

      if (content) {
        if (system_instruction) {
          system_instruction += '\n' + content;
        } else {
          system_instruction = content;
        }
      }
    } else {
      const turn = convertMessageToGoogleTurn(message);
      if (turn) {
        input.push(turn);
      }
    }
  }

  return { input, system_instruction };
}

/**
 * Parses the Google Interactions API response into LangChain ChatGenerations.
 */
export function convertInteractionToChatGeneration(
  response: Interactions.Interaction,
  usageMetadata?: UsageMetadata,
): ChatGeneration[] {
  const outputs = response.outputs || [];
  const contentBlocks: MessageContent = [];
  const toolCalls: ToolCall[] = [];
  const responseMetadata: Record<string, unknown> = {};

  // Extract usage if provided in the interaction object
  let usage: UsageMetadata | undefined = usageMetadata;
  if (!usage && response.usage) {
    usage = {
      input_tokens: response.usage.total_input_tokens || 0,
      output_tokens: response.usage.total_output_tokens || 0,
      total_tokens: response.usage.total_tokens || 0,
    };
  }

  let textContent = '';

  for (const output of outputs) {
    // Use bracket notation to access type to avoid potential TS union mismatches
    const type = output['type'];

    if (type === 'text') {
      const text = output['text'] || '';
      textContent += text;
      contentBlocks.push({ type: 'text', text: text });
    } else if (type === 'image') {
      // Handle generated images
      contentBlocks.push({
        type: 'image_url',
        image_url: {
          url: `data:${output['mime_type']};base64,${output['data']}`,
        },
      });
    } else if (type === 'function_call') {
      toolCalls.push({
        id: output['id'],
        name: output['name'],
        args: output['arguments'],
        type: 'tool_call',
      });
    } else if (type === 'thought') {
      // Capture thought signature for the next turn if available
      if (output['signature']) {
        responseMetadata['thought_signature'] = output['signature'];
      }
      // Handle reasoning content
      if (output.summary && output.summary.length > 0) {
        const reasoningText = output.summary.map(s => (s.type === 'text' ? s.text : '')).join('\n');
        contentBlocks.push({
          type: 'reasoning',
          reasoning: reasoningText,
        });
      }
    } else if (type === 'code_execution_result') {
      responseMetadata['code_execution_result'] = output;
    }
  }

  // Flatten content to string if it only contains text
  let finalContent: MessageContent = contentBlocks;
  const isOnlyText = contentBlocks.every(block => block.type === 'text');
  if (isOnlyText && textContent !== '') {
    finalContent = textContent;
  }

  const message = new AIMessage({
    content: finalContent.length > 0 ? finalContent : textContent,
    tool_calls: toolCalls,
    usage_metadata: usage,
    response_metadata: {
      interaction_id: response.id,
      finish_reason: response.status,
      model: response.model,
      ...responseMetadata,
    },
  });

  return [
    {
      text: textContent,
      message: message,
      generationInfo: {
        finish_reason: response.status,
      },
    },
  ];
}

/**
 * Converts a Google Interaction SSE Event into a LangChain AIMessageChunk.
 */
export function makeMessageChunkFromGoogleInteractionEvent(
  event: Interactions.InteractionSSEEvent,
): AIMessageChunk | null {
  if (!event || !('event_type' in event)) {
    return null;
  }

  if (event.event_type === 'content.delta') {
    const delta = event.delta;
    if (!delta) return null;

    const type = delta['type'];

    if (type === 'text') {
      return new AIMessageChunk({
        content: delta['text'] || '',
      });
    } else if (type === 'function_call') {
      return new AIMessageChunk({
        content: '',
        tool_call_chunks: [
          {
            index: event.index || 0,
            id: delta['id'],
            name: delta['name'],
            args: JSON.stringify(delta['arguments']),
            type: 'tool_call_chunk',
          },
        ],
      });
    } else if (type === 'thought_signature') {
      return new AIMessageChunk({
        content: '',
        response_metadata: {
          thought_signature: delta['signature'],
        },
      });
    } else if (type === 'thought_summary') {
      // If we receive a summary at the end, treat it as a reasoning block
      let text = '';
      if (delta['content'] && delta['content']['type'] === 'text') {
        text = delta['content']['text'] || '';
      }
      return new AIMessageChunk({
        content: [
          {
            type: 'reasoning',
            reasoning: text,
            index: event.index,
          },
        ],
      });
    }
  } else if (event.event_type === 'interaction.complete') {
    // Final event often contains usage
    const usage = event.interaction?.usage;
    if (usage) {
      return new AIMessageChunk({
        content: '',
        usage_metadata: {
          input_tokens: usage.total_input_tokens || 0,
          output_tokens: usage.total_output_tokens || 0,
          total_tokens: usage.total_tokens || 0,
        },
        response_metadata: {
          interaction_id: event.interaction?.id,
        },
      });
    }
  }

  return null;
}
