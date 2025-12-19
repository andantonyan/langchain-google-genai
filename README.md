# Experimental Google GenAI Integration for LangChain.js

This repository contains an experimental, custom implementation of a `ChatGoogleGenAI` class for LangChain.js. It was developed to address specific limitations and bugs in the official LangChain Google integrations, particularly concerning newer Gemini models.

This is an experiment and should not be considered production-ready. It was created as a proof-of-concept to resolve the issue outlined in [langchain-ai/langchainjs#9624](https://github.com/langchain-ai/langchainjs/issues/9624).

## Problem Solved

When using Gemini "Thinking" models (like `gemini-3-pro-preview`), a `thought_signature` is required when replying with tool outputs. The official LangChain libraries at the time of writing do not correctly handle this, leading to API errors. This implementation correctly captures and re-sends the `thought_signature`, enabling tool usage with these advanced models.

## Key Features

*   **Modern SDK**: Built on the new `@google/genai` SDK, not the deprecated `@google/generative-ai`.
*   **`thought_signature` Support**: Correctly handles the `thought_signature` required for tool calls with Gemini 3+ models.
*   **Experimental Interactions API**: Includes a functional implementation of the new [Google AI Interactions API](https://ai.google.dev/docs/interactions_api_overview), allowing for:
    *   Stateful, multi-turn conversations.
    *   Use of predefined Google Agents (e.g., `deep-research-pro-preview-12-2025`).
    *   Asynchronous operations for long-running tasks.
*   **Reasoning and Text Separation**: Properly distinguishes between the model's "thinking" or "reasoning" process and the final text output, returning them as distinct content blocks.
*   **Standard LangChain Compatibility**: Integrates as a `BaseChatModel` and supports standard features like `invoke`, `stream`, `bindTools`, and `withStructuredOutput`.

## Installation

```bash
npm install
```

## Configuration

Set your Google API key as an environment variable:

```bash
export GOOGLE_API_KEY="YOUR_API_KEY"
```

## Usage

The `ChatGoogleGenAI` class can be used as a drop-in replacement for other LangChain chat models.

### Basic Invocation

```typescript
import { ChatGoogleGenAI } from './src';

const model = new ChatGoogleGenAI({
  model: 'gemini-2.5-flash',
  temperature: 0.7,
});

const response = await model.invoke('Explain quantum entanglement in one sentence.');
console.log(response.content);
```

### Tool Calling with Gemini 3 (with `thought_signature`)

```typescript
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { ChatGoogleGenAI } from './src';

const model = new ChatGoogleGenAI({
  model: 'gemini-3-flash-preview',
  useExperimentalInteractionsApi: true, // Recommended for Gemini 3
});

const weatherTool = tool({
  name: 'get_weather',
  description: 'Get weather for a city',
  schema: z.object({ city: z.string() }),
  func: async ({ city }) => `The weather in ${city} is sunny.`,
});

const modelWithTools = model.bindTools([weatherTool]);

// The model will generate a tool call, and the subsequent
// response will correctly include the thought_signature.
const res = await modelWithTools.invoke('What is the weather in London?');
console.log('Tool Calls:', res.tool_calls);
```

### Stateful Conversation using Interactions API

```typescript
const model = new ChatGoogleGenAI({
  model: 'gemini-2.5-flash',
  useExperimentalInteractionsApi: true,
});

// Turn 1
const res1 = await model.invoke('Hi, my name is Phil.');
console.log(res1.content);

const interactionId = res1.response_metadata['interaction_id'];

// Turn 2 (maintains context)
const res2 = await model.invoke('What is my name?', {
  previousInteractionId: interactionId,
});
console.log(res2.content); // AI will remember your name
```

## Running Examples

The `examples.ts` file contains a comprehensive suite of tests covering all major features. To run it:

```bash
npm start
```
