import {
  type GoogleGenAIOptions,
  type HttpOptions,
  type Interactions,
  type SafetySetting,
  type Schema,
  type ThinkingConfig,
  type ToolConfig,
} from '@google/genai';
import {
  type BaseChatModelCallOptions,
  type BaseChatModelParams,
  type BindToolsInput,
} from '@langchain/core/language_models/chat_models';

/**
 * Input parameters for the ChatGoogleGenAI class.
 * Merges standard configuration with Interactions-specific fields.
 */
export interface ChatGoogleGenAIInput extends BaseChatModelParams {
  /**
   * Google API Key.
   */
  apiKey?: string;

  /**
   * The model name (e.g., "gemini-2.0-flash", "gemini-3-pro-preview").
   */
  model?: string;

  /**
   * The agent name (e.g., "deep-research-pro-preview-12-2025").
   * If provided, the model parameter is ignored in the API call.
   */
  agent?: string;

  /**
   * Client options passed directly to the Google GenAI Client.
   */
  clientOptions?: GoogleGenAIOptions;

  /**
   * Default temperature for the model.
   */
  temperature?: number;

  /**
   * Default max output tokens.
   */
  maxOutputTokens?: number;

  /**
   * Default topP.
   */
  topP?: number;

  /**
   * Default topK.
   */
  topK?: number;

  /**
   * Stop sequences.
   */
  stop?: string[];

  /**
   * Safety settings for the model.
   */
  safetySettings?: SafetySetting[];

  /**
   * Thinking config (for stable API).
   */
  thinkingConfig?: ThinkingConfig;

  /**
   * Whether to force the use of the Interactions API even for standard models.
   */
  useExperimentalInteractionsApi?: boolean;

  /**
   * API Version (e.g., "v1beta").
   */
  apiVersion?: string;

  /**
   * Google Cloud Project ID (for Vertex AI).
   */
  project?: string;

  /**
   * Google Cloud Location (for Vertex AI).
   */
  location?: string;

  /**
   * Whether to use Vertex AI.
   */
  vertexai?: boolean;

  /**
   * HTTP Options.
   */
  httpOptions?: HttpOptions;

  /**
   * Whether to stream the results.
   */
  streaming?: boolean;

  /**
   * Whether to include usage metadata in streaming chunks.
   */
  streamUsage?: boolean;

  /**
   * Whether to run the interaction in the background.
   * Only supported for agents.
   * Defaults to false.
   */
  background?: boolean;

  /**
   * Whether to store the interaction.
   * Defaults to true. Set to false to opt out of storage.
   * Note: store=false is incompatible with background=true and prevents using previousInteractionId.
   */
  store?: boolean;
}

/**
 * Call options for the ChatGoogleGenAI model.
 * Supports both standard generateContent options and Interactions API options.
 */
export interface ChatGoogleGenAICallOptions extends BaseChatModelCallOptions {
  /**
   * Tools to bind to the model. Can be LangChain tools or Google built-in tool definitions.
   */
  tools?: (Interactions.Tool | BindToolsInput)[];

  /**
   * Tool choice configuration.
   */
  tool_choice?: Interactions.ToolChoice | ToolConfig;

  /**
   * The ID of the previous interaction to continue a stateful conversation (Interactions API).
   */
  previousInteractionId?: string;

  /**
   * Whether to run the interaction in the background (Interactions API).
   * Note: Only supported for agents.
   */
  background?: boolean;

  /**
   * Response MIME type (Standard API).
   */
  responseMimeType?: string;

  /**
   * Response Schema (Standard API).
   */
  responseSchema?: Schema;

  /**
   * Tool configuration (Standard API).
   */
  toolConfig?: ToolConfig;
}
