import {
  FunctionCallingConfigMode,
  type FunctionDeclaration,
  type GenerateContentConfig,
  GoogleGenAI,
  type GoogleGenAIOptions,
  type Interactions,
  type SafetySetting,
  type Schema,
  type ThinkingConfig,
  type Tool,
  type ToolConfig,
} from '@google/genai';
import { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import { type BaseLanguageModelInput, type StructuredOutputMethodOptions } from '@langchain/core/language_models/base';
import { BaseChatModel, BindToolsInput, LangSmithParams } from '@langchain/core/language_models/chat_models';
import { AIMessageChunk, type BaseMessage } from '@langchain/core/messages';
import { JsonOutputParser } from '@langchain/core/output_parsers';
import { ChatGenerationChunk, type ChatResult } from '@langchain/core/outputs';
import { Runnable, RunnableSequence } from '@langchain/core/runnables';
import { isStructuredTool } from '@langchain/core/tools';
import { getEnvironmentVariable } from '@langchain/core/utils/env';
import { toJsonSchema } from '@langchain/core/utils/json_schema';
import { type InteropZodType, isInteropZodSchema } from '@langchain/core/utils/types';

import { ChatGoogleGenAICallOptions, ChatGoogleGenAIInput } from './types.js';
import {
  convertInteractionToChatGeneration,
  convertMessagesToGoogleInteractionPayload,
  makeMessageChunkFromGoogleInteractionEvent,
} from './utils/interaction-utils.js';
import { convertMessagesToGooglePayload } from './utils/message-inputs.js';
import {
  convertGoogleResponseToChatGeneration,
  convertGoogleStreamChunkToLangChainChunk,
} from './utils/message-outputs.js';

/**
 * Google Gemini Chat Model integration.
 */
export class ChatGoogleGenAI extends BaseChatModel<ChatGoogleGenAICallOptions> {
  static override lc_name() {
    return 'ChatGoogleGenAI';
  }

  override get lc_secrets(): { [key: string]: string } | undefined {
    return {
      apiKey: 'GOOGLE_API_KEY',
    };
  }

  model?: string;

  apiKey?: string;

  clientOptions?: GoogleGenAIOptions;

  temperature?: number;

  maxOutputTokens?: number;

  topP?: number;

  topK?: number;

  stop?: string[];

  safetySettings?: SafetySetting[];

  thinkingConfig?: ThinkingConfig;

  streamUsage = true;

  streaming = false;

  useExperimentalInteractionsApi: boolean;

  agent?: string;

  background?: boolean;

  store?: boolean;

  get requiredAgent(): string {
    if (!this.agent) {
      throw new Error('This operation requires an agent to be specified.');
    }
    return this.agent;
  }

  get requiredModel(): string {
    if (!this.model) {
      throw new Error('This operation requires a model to be specified.');
    }
    return this.model;
  }

  private client: GoogleGenAI;

  constructor(fields?: ChatGoogleGenAIInput) {
    super(fields ?? {});

    if (fields?.model && fields?.agent) {
      throw new Error('Both "model" and "agent" cannot be specified. Please provide only one.');
    }

    if (!fields?.model && !fields?.agent) {
      throw new Error('Either "model" or "agent" must be specified in the constructor fields.');
    }

    this.model = fields?.model;
    this.agent = fields?.agent;
    this.background = fields?.background;
    this.store = fields?.store;

    if (this.store === false && this.background === true) {
      throw new Error('Invalid configuration: "store" cannot be false when "background" is true.');
    }

    if (this.agent) {
      if (fields?.useExperimentalInteractionsApi === false) {
        throw new Error('When "agent" is specified, "useExperimentalInteractionsApi" must be true.');
      }
      // Agents implicitly use interactions API
      this.useExperimentalInteractionsApi = true;
    } else {
      this.useExperimentalInteractionsApi = fields?.useExperimentalInteractionsApi ?? false;
    }

    this.apiKey = fields?.apiKey ?? getEnvironmentVariable('GOOGLE_API_KEY');

    if (!this.apiKey) {
      throw new Error(
        'Google API key not found. Please set the GOOGLE_API_KEY environment variable or pass it to the constructor.',
      );
    }

    this.clientOptions = {
      apiKey: this.apiKey,
      apiVersion: fields?.apiVersion,
      httpOptions: fields?.httpOptions,
      project: fields?.project,
      location: fields?.location,
      vertexai: fields?.vertexai,
    };

    this.temperature = fields?.temperature;
    this.maxOutputTokens = fields?.maxOutputTokens;
    this.topP = fields?.topP;
    this.topK = fields?.topK;
    this.stop = fields?.stop;
    this.safetySettings = fields?.safetySettings;
    this.thinkingConfig = fields?.thinkingConfig;
    this.streaming = fields?.streaming ?? this.streaming;
    this.streamUsage = fields?.streamUsage ?? this.streamUsage;

    this.client = new GoogleGenAI(this.clientOptions);
  }

  _llmType() {
    return 'google_genai';
  }

  override getLsParams(options: this['ParsedCallOptions']): LangSmithParams {
    return {
      ls_provider: 'google_genai',
      ls_model_name: this.model,
      ls_model_type: 'chat',
      ls_temperature: this.temperature,
      ls_max_tokens: this.maxOutputTokens,
      ls_stop: options.stop,
    };
  }

  /**
   * Main generation method. Branches between Standard and Interactions API.
   */
  async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    if (this.shouldUseInteractionsApi(options)) {
      return this.generateInteractions(messages, options, runManager);
    }

    if (this.streaming) {
      const stream = this._streamResponseChunks(messages, options, runManager);
      let finalChunk: ChatGenerationChunk | undefined;
      for await (const chunk of stream) {
        if (!finalChunk) {
          finalChunk = chunk;
        } else {
          finalChunk = finalChunk.concat(chunk);
        }
      }
      if (finalChunk === undefined) {
        throw new Error('No chunks returned from Google GenAI.');
      }
      return {
        generations: [
          {
            text: finalChunk.text,
            message: finalChunk.message,
          },
        ],
      };
    }

    const params = this.invocationParams(options);
    const { contents, systemInstruction } = convertMessagesToGooglePayload(messages);
    const tools = this.formatTools(options.tools);
    const toolConfig = this.formatToolConfig(options);

    const config: GenerateContentConfig = {
      ...params,
      systemInstruction: systemInstruction?.parts,
      tools,
      toolConfig,
    };

    const response = await this.client.models.generateContent({
      model: this.requiredModel,
      contents,
      config,
    });

    const generation = convertGoogleResponseToChatGeneration(response);
    return { generations: [generation] };
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    if (this.shouldUseInteractionsApi(options)) {
      const { input, system_instruction } = convertMessagesToGoogleInteractionPayload(messages);

      let stream: AsyncIterable<Interactions.InteractionSSEEvent>;

      if (this.agent) {
        stream = (await this.client.interactions.create(
          this.interactiveAgentInvocationParams(input, options, true),
        )) as AsyncIterable<Interactions.InteractionSSEEvent>;
      } else {
        stream = (await this.client.interactions.create(
          this.interactiveInvocationParams(input, system_instruction, options, true),
        )) as AsyncIterable<Interactions.InteractionSSEEvent>;
      }

      for await (const event of stream) {
        const chunk = makeMessageChunkFromGoogleInteractionEvent(event);
        if (chunk) {
          const generationChunk = new ChatGenerationChunk({
            message: chunk,
            text: chunk.text ?? '',
          });
          yield generationChunk;
          await runManager?.handleLLMNewToken(chunk.text ?? '', undefined, undefined, undefined, undefined, {
            chunk: generationChunk,
          });
        }
      }
      return;
    }

    // Standard API stream
    const params = this.invocationParams(options);
    const { contents, systemInstruction } = convertMessagesToGooglePayload(messages);
    const tools = this.formatTools(options.tools);
    const toolConfig = this.formatToolConfig(options);

    const config: GenerateContentConfig = {
      ...params,
      systemInstruction: systemInstruction?.parts,
      tools,
      toolConfig,
    };

    const stream = await this.client.models.generateContentStream({
      model: this.requiredModel,
      contents,
      config,
    });

    for await (const chunk of stream) {
      const generationChunk = convertGoogleStreamChunkToLangChainChunk(chunk);
      if (generationChunk) {
        yield generationChunk;
        await runManager?.handleLLMNewToken(generationChunk.text ?? '', undefined, undefined, undefined, undefined, {
          chunk: generationChunk,
        });
      }
    }
  }

  override invocationParams(options?: this['ParsedCallOptions']): GenerateContentConfig {
    return {
      candidateCount: 1,
      stopSequences: options?.stop ?? this.stop,
      maxOutputTokens: this.maxOutputTokens,
      temperature: this.temperature,
      topP: this.topP,
      topK: this.topK,
      safetySettings: this.safetySettings,
      thinkingConfig: this.thinkingConfig,
      responseMimeType: options?.responseMimeType,
      responseSchema: options?.responseSchema,
    };
  }

  private convertToInteractionGenerationConfig(config: GenerateContentConfig): Interactions.GenerationConfig {
    let thinkingLevel: Interactions.ThinkingLevel | undefined;

    if (config.thinkingConfig?.thinkingLevel) {
      if (config.thinkingConfig.thinkingLevel === 'THINKING_LEVEL_UNSPECIFIED') {
        thinkingLevel = undefined;
      } else {
        thinkingLevel = config.thinkingConfig.thinkingLevel.toLowerCase() as Interactions.ThinkingLevel;
      }
    }

    return {
      temperature: config.temperature,
      max_output_tokens: config.maxOutputTokens,
      top_p: config.topP,
      // top_k: config.topK, // Not supported
      stop_sequences: config.stopSequences,
      thinking_level: thinkingLevel,
      thinking_summaries: config.thinkingConfig?.includeThoughts ? 'auto' : undefined,
    };
  }

  override bindTools(
    tools: ChatGoogleGenAICallOptions['tools'],
    kwargs?: Partial<ChatGoogleGenAICallOptions>,
  ): Runnable<BaseLanguageModelInput, AIMessageChunk, ChatGoogleGenAICallOptions> {
    return this.withConfig({
      tools,
      ...kwargs,
    });
  }

  override withStructuredOutput<
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    RunOutput extends Record<string, any> = Record<string, any>,
  >(
    outputSchema:
      | InteropZodType<RunOutput>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      | Record<string, any>,
    config?: StructuredOutputMethodOptions<false>,
  ): Runnable<BaseLanguageModelInput, RunOutput>;

  override withStructuredOutput<
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    RunOutput extends Record<string, any> = Record<string, any>,
  >(
    outputSchema:
      | InteropZodType<RunOutput>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      | Record<string, any>,
    config?: StructuredOutputMethodOptions<true>,
  ): Runnable<BaseLanguageModelInput, { raw: BaseMessage; parsed: RunOutput }>;

  override withStructuredOutput<
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    RunOutput extends Record<string, any> = Record<string, any>,
  >(
    outputSchema:
      | InteropZodType<RunOutput>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      | Record<string, any>,
    config?: StructuredOutputMethodOptions<boolean>,
  ):
    | Runnable<BaseLanguageModelInput, RunOutput>
    | Runnable<BaseLanguageModelInput, { raw: BaseMessage; parsed: RunOutput }> {
    const schema = outputSchema;
    const method = config?.method;
    const includeRaw = config?.includeRaw;

    if (method === 'jsonMode') {
      throw new Error("Google GenAI does not support 'jsonMode'. Use 'jsonSchema' or 'functionCalling'.");
    }

    let llm: Runnable<BaseLanguageModelInput>;
    let outputParser: Runnable<BaseMessage, RunOutput>;

    if (method === 'jsonSchema' || method === undefined) {
      // Use Google's native JSON schema support
      const jsonSchema = isInteropZodSchema(schema) ? toJsonSchema(schema) : schema;

      llm = this.withConfig({
        responseMimeType: 'application/json',
        responseSchema: jsonSchema as Schema,
      });

      outputParser = new JsonOutputParser<RunOutput>();
    } else {
      throw new Error(`Unrecognized structured output method '${method}'. Google GenAI supports 'jsonSchema'.`);
    }

    if (includeRaw) {
      return RunnableSequence.from([
        {
          raw: llm,
        },
        {
          raw: input => input.raw,
          parsed: input => outputParser.invoke(input.raw),
        },
      ]);
    }

    return llm.pipe(outputParser);
  }

  /**
   * Public method to retrieve an interaction's status and results.
   */
  async getInteraction(interactionId: string): Promise<BaseMessage> {
    const interaction = await this.client.interactions.get(interactionId);
    const [generation] = convertInteractionToChatGeneration(interaction);
    return generation.message;
  }

  private formatToolConfig(options: this['ParsedCallOptions']): ToolConfig | undefined {
    if (options.toolConfig) return options.toolConfig;

    if (options.tool_choice) {
      if (typeof options.tool_choice === 'string') {
        const modeMap: Record<string, FunctionCallingConfigMode> = {
          AUTO: FunctionCallingConfigMode.AUTO,
          ANY: FunctionCallingConfigMode.ANY,
          NONE: FunctionCallingConfigMode.NONE,
        };
        const choice = options.tool_choice.toUpperCase();
        if (choice in modeMap) {
          return {
            functionCallingConfig: {
              mode: modeMap[choice],
            },
          };
        }
      } else if (typeof options.tool_choice === 'object') {
        // Handle the case where tool_choice is already a ToolConfig-compatible object
        const toolChoiceObj = options.tool_choice;
        if ('functionCallingConfig' in toolChoiceObj || 'mode' in toolChoiceObj) {
          return {
            functionCallingConfig: toolChoiceObj as { mode: FunctionCallingConfigMode },
          };
        }
      }
    }
    return undefined;
  }

  private formatTools(tools?: ChatGoogleGenAICallOptions['tools']): Tool[] | undefined {
    if (!tools) return undefined;
    const toolList = Array.isArray(tools) ? tools : [tools];
    if (toolList.length === 0) return undefined;

    const functionDeclarations: FunctionDeclaration[] = [];
    const googleTools: Tool[] = [];

    for (const tool of toolList) {
      // Check if it's a Google Interaction Tool (Tool_2) or standard Tool
      if ('functionDeclarations' in tool || 'googleSearch' in tool || 'codeExecution' in tool) {
        googleTools.push(tool as Tool);
        continue;
      }
      const lcTool = tool as BindToolsInput;
      if (isStructuredTool(lcTool)) {
        const schema = isInteropZodSchema(lcTool.schema) ? toJsonSchema(lcTool.schema) : lcTool.schema;
        functionDeclarations.push({
          name: lcTool.name,
          description: lcTool.description,
          parameters: schema as Schema,
        });
      }
    }

    if (functionDeclarations.length > 0) {
      googleTools.push({ functionDeclarations });
    }

    return googleTools.length > 0 ? googleTools : undefined;
  }

  private formatInteractionToolConfig(options: this['ParsedCallOptions']): Interactions.ToolChoice | undefined {
    if (options.tool_choice) {
      if (typeof options.tool_choice === 'string') {
        const modeMap: Record<string, string> = {
          AUTO: 'auto',
          ANY: 'any',
          NONE: 'none',
        };
        const choice = options.tool_choice.toUpperCase();
        if (choice in modeMap) {
          return {
            allowed_tools: {
              mode: modeMap[choice] as 'auto' | 'any' | 'none',
            },
          };
        }
      } else if (typeof options.tool_choice === 'object') {
        // Assume compatible object
        return options.tool_choice as Interactions.ToolChoice;
      }
    }
    // Map toolConfig to Interactions tool choice if possible
    if (options.toolConfig?.functionCallingConfig) {
      const mode = options.toolConfig.functionCallingConfig.mode;
      const modeString =
        mode === FunctionCallingConfigMode.AUTO ? 'auto' : mode === FunctionCallingConfigMode.ANY ? 'any' : 'none';
      return {
        allowed_tools: {
          mode: modeString,
        },
      };
    }
    return undefined;
  }

  private formatInteractionTools(tools?: ChatGoogleGenAICallOptions['tools']): Interactions.Tool[] | undefined {
    if (!tools) return undefined;
    const toolList = Array.isArray(tools) ? tools : [tools];
    if (toolList.length === 0) return undefined;

    const googleTools: Interactions.Tool[] = [];

    for (const tool of toolList) {
      // Check if it's already a Google Interaction Tool (snake_case)
      // Tool_2 is a union of specific interfaces, so we check for 'type' or properties
      if (
        'type' in tool &&
        (tool.type === 'function' || tool.type === 'google_search' || tool.type === 'code_execution')
      ) {
        googleTools.push(tool as Interactions.Tool);
        continue;
      }

      // Check if it's a Standard Tool (camelCase) and convert
      if ('functionDeclarations' in tool) {
        // Convert Standard FunctionDeclarations to Interaction Format (Function_2)
        const funcs = (tool as Tool).functionDeclarations;
        if (funcs) {
          for (const func of funcs) {
            googleTools.push({
              type: 'function',
              name: func.name,
              description: func.description,
              parameters: func.parameters as Schema,
            });
          }
        }
        // Handle other standard tools if needed (googleSearch, etc.)
        if ('googleSearch' in tool) {
          googleTools.push({ type: 'google_search' });
        }
        if ('codeExecution' in tool) {
          googleTools.push({ type: 'code_execution' });
        }
        continue;
      }

      const lcTool = tool as BindToolsInput;
      if (isStructuredTool(lcTool)) {
        const schema = isInteropZodSchema(lcTool.schema) ? toJsonSchema(lcTool.schema) : lcTool.schema;
        googleTools.push({
          type: 'function',
          name: lcTool.name,
          description: lcTool.description,
          parameters: schema as Schema,
        });
      }
    }

    return googleTools.length > 0 ? googleTools : undefined;
  }

  private interactiveAgentInvocationParams(
    input: Interactions.Turn[],
    options: this['ParsedCallOptions'],
    stream: boolean,
  ) {
    return {
      agent: this.requiredAgent,
      background: options.background ?? this.background,
      stream,
      input,
      previous_interaction_id: options.previousInteractionId,
      store: this.store,
    };
  }

  private interactiveInvocationParams(
    input: Interactions.Turn[],
    system_instruction: string | undefined,
    options: this['ParsedCallOptions'],
    stream: boolean,
  ) {
    const params = this.invocationParams(options);
    const tools = this.formatInteractionTools(options.tools);
    const tool_choice = this.formatInteractionToolConfig(options);
    const generation_config = this.convertToInteractionGenerationConfig(params);

    return {
      model: this.requiredModel,
      input,
      system_instruction,
      previous_interaction_id: options.previousInteractionId,
      tools,
      generation_config: {
        ...generation_config,
        tool_choice,
      },
      response_mime_type: params.responseMimeType,
      response_format: params.responseSchema,
      background: options.background ?? this.background,
      stream,
      store: this.store,
    };
  }

  /**
   * Determines if the Interactions API should be used based on call options.
   */
  private shouldUseInteractionsApi(options: this['ParsedCallOptions']): boolean {
    return !!(this.useExperimentalInteractionsApi || options.previousInteractionId);
  }

  private async generateInteractions(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    const { input, system_instruction } = convertMessagesToGoogleInteractionPayload(messages);

    if (this.streaming) {
      const stream = this._streamResponseChunks(messages, options, runManager);
      let finalChunk: ChatGenerationChunk | undefined;
      for await (const chunk of stream) {
        if (!finalChunk) {
          finalChunk = chunk;
        } else {
          finalChunk = finalChunk.concat(chunk);
        }
      }
      if (!finalChunk) {
        throw new Error('No chunks from Interactions API');
      }
      return { generations: [{ text: finalChunk.text, message: finalChunk.message }] };
    }

    let interaction: Interactions.Interaction;

    if (this.agent) {
      interaction = (await this.client.interactions.create(
        this.interactiveAgentInvocationParams(input, options, false),
      )) as Interactions.Interaction;
    } else {
      interaction = (await this.client.interactions.create(
        this.interactiveInvocationParams(input, system_instruction, options, false),
      )) as Interactions.Interaction;
    }

    const generations = convertInteractionToChatGeneration(interaction);
    return { generations };
  }
}
