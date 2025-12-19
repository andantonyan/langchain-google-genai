import { type Content, type Part } from '@google/genai';
import { AIMessage, type BaseMessage, type MessageContent, SystemMessage, ToolMessage } from '@langchain/core/messages';
import { type ToolCall } from '@langchain/core/messages/tool';

/**
 * Helper to parse a base64 data URL into mimeType and data.
 */
function parseBase64Data(dataUrl: string): { mimeType: string; data: string } {
  const matches = dataUrl.match(/^data:(.+);base64,(.+)$/);
  if (!matches || matches.length !== 3) {
    throw new Error('Invalid base64 image data URL');
  }
  return { mimeType: matches[1], data: matches[2] };
}

/**
 * Converts a LangChain MessageContent (string or complex array) into Google Parts.
 */
function convertContentToParts(content: MessageContent): Part[] {
  if (typeof content === 'string') {
    if (content === '') return [];
    return [{ text: content }];
  }

  return content.map((block): Part => {
    const b = block as {
      type: string;
      text?: string;
      image_url?: string | { url: string };
      reasoning?: string;
    };

    if (b.type === 'text' && typeof b.text === 'string') {
      return { text: b.text };
    } else if (b.type === 'reasoning' && typeof b.reasoning === 'string') {
      // Convert LangChain reasoning block back to Google thought part
      return { text: b.reasoning, thought: true };
    } else if (b.type === 'image_url' && b.image_url) {
      let url: string;
      if (typeof b.image_url === 'string') {
        url = b.image_url;
      } else if (typeof b.image_url === 'object' && 'url' in b.image_url) {
        url = b.image_url.url;
      } else {
        throw new Error('Invalid image_url block format');
      }

      // Handle Base64
      if (url.startsWith('data:')) {
        const { mimeType, data } = parseBase64Data(url);
        return {
          inlineData: {
            mimeType,
            data,
          },
        };
      }
      // Handle File URI (Google Cloud Storage or File API)
      else if (url.startsWith('gs://') || url.startsWith('https://')) {
        return {
          fileData: {
            mimeType: 'image/jpeg', // Fallback, ideally should be inferred
            fileUri: url,
          },
        };
      }
    }
    throw new Error(`Unsupported content block type: ${b.type}`);
  });
}

/**
 * Converts a LangChain ToolCall to a Google FunctionCall Part.
 */
function convertToolCallToPart(toolCall: ToolCall): Part {
  return {
    functionCall: {
      name: toolCall.name,
      args: toolCall.args,
    },
  };
}

/**
 * Converts a LangChain ToolMessage to a Google FunctionResponse Part.
 */
function convertToolMessageToPart(message: ToolMessage): Part {
  return {
    functionResponse: {
      name: message.name || '', // Name is required by Google
      response: {
        name: message.name,
        content: message.content,
      },
    },
  };
}

/**
 * Main function to convert LangChain messages to Google GenAI payload.
 * Handles merging consecutive messages of the same role.
 */
export function convertMessagesToGooglePayload(messages: BaseMessage[]): {
  contents: Content[];
  systemInstruction?: Content;
} {
  const contents: Content[] = [];
  let systemInstruction: Content | undefined;

  // Extract System Messages
  const systemMessages = messages.filter(msg => SystemMessage.isInstance(msg));
  if (systemMessages.length > 0) {
    const systemParts = systemMessages.flatMap(msg => convertContentToParts(msg.content));
    if (systemParts.length > 0) {
      systemInstruction = {
        role: 'system', // Technically 'system' isn't a valid role in 'contents', but used for config
        parts: systemParts,
      };
    }
  }

  // Process Non-System Messages
  const chatMessages = messages.filter(msg => !SystemMessage.isInstance(msg));

  for (const message of chatMessages) {
    const role = AIMessage.isInstance(message) ? 'model' : 'user';
    const parts: Part[] = [];

    // Handle Content (Text/Images/Reasoning)
    parts.push(...convertContentToParts(message.content));

    // Handle Tool Calls (AI Message)
    if (AIMessage.isInstance(message) && message.tool_calls?.length) {
      for (const toolCall of message.tool_calls) {
        parts.push(convertToolCallToPart(toolCall));
      }
    }

    // Handle Tool Results (Tool Message)
    // Note: ToolMessages are mapped to 'user' role in Google GenAI
    if (ToolMessage.isInstance(message)) {
      parts.push(convertToolMessageToPart(message));
    }

    // Reinject Thought Signature if present in AIMessage metadata
    if (AIMessage.isInstance(message)) {
      // Safely access response_metadata by casting to a Record
      const metadata = message.response_metadata as Record<string, unknown>;

      if (metadata && typeof metadata['thoughtSignature'] === 'string') {
        const signature = metadata['thoughtSignature'];

        if (parts.length > 0) {
          parts[parts.length - 1].thoughtSignature = signature;
        } else {
          // If no content (e.g. pure tool call message), attach to a dummy text part
          parts.push({ text: '', thoughtSignature: signature });
        }
      }
    }

    // Merge logic: If the last message in `contents` has the same role, append parts.
    const lastContent = contents[contents.length - 1];

    if (lastContent && lastContent.role === role) {
      if (!lastContent.parts) {
        lastContent.parts = [];
      }
      lastContent.parts.push(...parts);
    } else {
      contents.push({
        role,
        parts,
      });
    }
  }

  return {
    contents,
    systemInstruction,
  };
}
