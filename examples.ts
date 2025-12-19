import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { Calculator } from '@langchain/community/tools/calculator';
import { createAgent } from 'langchain';

import { ChatGoogleGenAI } from './src';

// Ensure API key is set
process.env.GOOGLE_API_KEY = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY || '';

// Configuration for models
const MODELS = ['gemini-2.5-flash', 'gemini-3-flash-preview'];

/**
 * Helper to run a test for multiple models
 */
async function runForModels(testName: string, fn: (modelName: string) => Promise<void>) {
  console.log(`

==================================================`);
  console.log(`RUNNING SUITE: ${testName}`);
  console.log(`==================================================`);

  for (const model of MODELS) {
    console.log(`
--- Testing Model: ${model} ---`);
    try {
      await fn(model);
    } catch (e) {
      console.error(`FAILED [${model}]:`, e);
    }
  }
}

/**
 * 1. Standard API: Basic Invoke & Stream
 */
async function testStandardBasic() {
  await runForModels('Standard API: Invoke & Stream', async modelName => {
    const model = new ChatGoogleGenAI({
      model: modelName,
      temperature: 0.7,
    });

    // Invoke
    console.log('[Invoke] User: Explain quantum entanglement in one sentence.');
    const response = await model.invoke('Explain quantum entanglement in one sentence.');
    console.log('[Invoke] AI:', response.content);

    // Stream
    console.log('\n[Stream] User: Count to 5.');
    const stream = await model.stream('Count to 5.');
    process.stdout.write('[Stream] AI: ');
    for await (const chunk of stream) {
      if (typeof chunk.content === 'string') {
        process.stdout.write(chunk.content);
      }
    }
    console.log('');
  });
}

/**
 * 2. Interactions API: Basic Invoke & Stream (Stateful)
 */
async function testInteractionsBasic() {
  await runForModels('Interactions API: Invoke & Stream', async modelName => {
    const model = new ChatGoogleGenAI({
      model: modelName,
      useExperimentalInteractionsApi: true,
    });

    // Invoke (Turn 1)
    console.log('[Invoke] User: Hi, my name is Phil.');
    const res1 = await model.invoke('Hi, my name is Phil.');
    console.log('[Invoke] AI:', res1.content);

    const interactionId = res1.response_metadata['interaction_id'] as string;
    console.log(`[State] Session ID: ${interactionId}`);

    if (!interactionId) {
      throw new Error('No interaction_id returned.');
    }

    // Invoke (Turn 2 - Stateful)
    console.log('[Invoke] User: What is my name? (using previousInteractionId)');
    const res2 = await model.invoke('What is my name?', {
      previousInteractionId: interactionId,
    });
    console.log('[Invoke] AI:', res2.content);

    // Stream (Turn 3 - Stateful)
    console.log('\n[Stream] User: Tell me a short joke about my name.');
    const stream = await model.stream('Tell me a short joke about my name.', {
      previousInteractionId: interactionId, // Continue same session
    });

    process.stdout.write('[Stream] AI: ');
    for await (const chunk of stream) {
      if (typeof chunk.content === 'string') {
        process.stdout.write(chunk.content);
      }
    }
    console.log('');
  });
}

/**
 * 3. Tools: Standard API
 */
async function testStandardTools() {
  await runForModels('Standard API: Tools', async modelName => {
    const model = new ChatGoogleGenAI({model: modelName});

    const weatherTool = tool(async ({city}) => `The weather in ${city} is sunny.`, {
      name: 'get_weather',
      schema: z.object({city: z.string()}),
      description: 'Get weather for a city',
    });

    const modelWithTools = model.bindTools([weatherTool]);

    console.log('User: What is the weather in Paris?');
    const res = await modelWithTools.invoke('What is the weather in Paris?');

    console.log('Tool Calls:', JSON.stringify(res.tool_calls, null, 2));
  });
}

/**
 * 4. Tools: Interactions API
 */
async function testInteractionsTools() {
  await runForModels('Interactions API: Tools', async modelName => {
    const model = new ChatGoogleGenAI({
      model: modelName,
      useExperimentalInteractionsApi: true,
    });

    const weatherTool = tool(async ({city}) => `The weather in ${city} is sunny.`, {
      name: 'get_weather',
      schema: z.object({city: z.string()}),
      description: 'Get weather for a city',
    });

    const modelWithTools = model.bindTools([weatherTool]);

    console.log('User: What is the weather in London?');
    const res = await modelWithTools.invoke('What is the weather in London?');

    console.log('Tool Calls:', JSON.stringify(res.tool_calls, null, 2));

    const interactionId = res.response_metadata['interaction_id'];
    console.log(`[State] Session ID: ${interactionId}`);
  });
}

/**
 * 5. Gemini 3 Features (Thoughts) - Standard & Interaction
 */
async function testGemini3Thoughts() {
  // Only run for gemini-3
  const modelName = 'gemini-3-flash-preview';
  console.log(`

==================================================`);
  console.log(`RUNNING SUITE: Gemini 3 Thoughts (${modelName})`);
  console.log(`==================================================`);

  // 1. Standard API Test
  console.log('\n--- Standard API: Thinking ---');
  const standardModel = new ChatGoogleGenAI({
    model: modelName,
    useExperimentalInteractionsApi: false,
    thinkingConfig: {includeThoughts: true},
  });

  console.log('User: Explain why the sky is blue.');
  const stdRes = await standardModel.invoke('Explain why the sky is blue.');

  // Check content for reasoning blocks
  if (Array.isArray(stdRes.content)) {
    const reasoning = stdRes.content.find(c => c.type === 'reasoning');
    console.log('[Standard] Has Reasoning Content:', !!reasoning);
    if (reasoning && 'reasoning' in reasoning) {
      console.log('[Standard] Reasoning snippet:', (reasoning.reasoning as string).substring(0, 100) + '...');
    }
  } else {
    console.log('[Standard] Content is string (no reasoning block returned in content array).');
  }

  // 2. Interactions API Test
  console.log('\n--- Interactions API: Thinking ---');
  const model = new ChatGoogleGenAI({
    model: modelName,
    useExperimentalInteractionsApi: true,
    thinkingConfig: {includeThoughts: true},
  });

  console.log('User: Explain why the sky is blue.');
  const res = await model.invoke('Explain why the sky is blue.');

  // Check metadata for signature
  const signature = res.response_metadata['thought_signature'];
  console.log('[Interactions] Thought Signature present:', !!signature);

  // Check content for reasoning blocks
  if (Array.isArray(res.content)) {
    const reasoning = res.content.find(c => c.type === 'reasoning');
    console.log('[Interactions] Has Reasoning Content:', !!reasoning);
    if (reasoning && 'reasoning' in reasoning) {
      console.log('[Interactions] Reasoning snippet:', (reasoning.reasoning as string).substring(0, 100) + '...');
    }
  } else {
    console.log('[Interactions] Content is string (no reasoning block returned in content array).');
  }
}

/**
 * 6. Gemini 2.5 Thoughts - Standard & Interaction
 */
async function testGemini25Thoughts() {
  const modelName = 'gemini-2.5-flash';
  console.log(`\n\n==================================================`);
  console.log(`RUNNING SUITE: Gemini 2.5 Thoughts (${modelName})`);
  console.log(`==================================================`);

  // 1. Standard API Test
  console.log('\n--- Standard API: Thinking ---');
  const standardModel = new ChatGoogleGenAI({
    model: modelName,
    useExperimentalInteractionsApi: false,
    thinkingConfig: {includeThoughts: true},
  });

  console.log('User: Explain why the sky is blue.');
  const stdRes = await standardModel.invoke('Explain why the sky is blue.');

  // Check content for reasoning blocks
  if (Array.isArray(stdRes.content)) {
    const reasoning = stdRes.content.find(c => c.type === 'reasoning');
    console.log(`[Standard ${modelName}] Has Reasoning Content:`, !!reasoning);
    if (reasoning && 'reasoning' in reasoning) {
      console.log(
        `[Standard ${modelName}] Reasoning snippet:`,
        (reasoning.reasoning as string).substring(0, 100) + '...',
      );
    }
  } else {
    console.log(`[Standard ${modelName}] Content is string (no reasoning block returned in content array).`);
  }

  // 2. Interactions API Test
  console.log('\n--- Interactions API: Thinking ---');
  const model = new ChatGoogleGenAI({
    model: modelName,
    useExperimentalInteractionsApi: true,
    thinkingConfig: {includeThoughts: true},
  });

  console.log('User: Explain why the sky is blue.');
  const res = await model.invoke('Explain why the sky is blue.');

  // Check metadata for signature
  const signature = res.response_metadata['thought_signature'];
  console.log(`[Interactions ${modelName}] Thought Signature present:`, !!signature);

  // Check content for reasoning blocks
  if (Array.isArray(res.content)) {
    const reasoning = res.content.find(c => c.type === 'reasoning');
    console.log(`[Interactions ${modelName}] Has Reasoning Content:`, !!reasoning);
    if (reasoning && 'reasoning' in reasoning) {
      console.log(
        `[Interactions ${modelName}] Reasoning snippet:`,
        (reasoning.reasoning as string).substring(0, 100) + '...',
      );
    }
  } else {
    console.log(`[Interactions ${modelName}] Content is string (no reasoning block returned in content array).`);
  }
}

/**
 * 7. Structured Output
 */
async function testStructuredOutput() {
  console.log('\n\n==================================================');
  console.log('RUNNING SUITE: Structured Output');
  console.log('==================================================');

  const model = new ChatGoogleGenAI({
    model: 'gemini-2.5-flash',
    temperature: 0,
  });

  const calculatorSchema = z.object({
    operation: z.enum(['add', 'subtract', 'multiply', 'divide']).describe('The operation to perform'),
    number1: z.number().describe('The first number'),
    number2: z.number().describe('The second number'),
  });

  const modelWithStructure = model.withStructuredOutput(calculatorSchema);

  console.log('User: What is 5 times 3?');
  const res = await modelWithStructure.invoke('What is 5 times 3?');

  console.log('Structured Output:', JSON.stringify(res, null, 2));
}

/**
 * 8. Deep Research Agent (Async/Manual Polling)
 * This test is defined but NOT run by default as requested.
 */
// eslint-disable-next-line @typescript-eslint/no-unused-vars
async function testDeepResearchAgent() {
  console.log('\n\n==================================================');
  console.log('TEST: Deep Research Agent');
  console.log('==================================================');

  const agent = new ChatGoogleGenAI({
    agent: 'deep-research-pro-preview-12-2025',
  });

  console.log('1. Starting Research...');
  const initialResponse = await agent.invoke('Research the history of Google TPUs with a focus on 2025 specs.');

  const interactionId = initialResponse.response_metadata['interaction_id'] as string;
  console.log(`   Job Started. Interaction ID: ${interactionId}`);

  // 2. Polling for results
  console.log('2. Polling for results...');

  await new Promise<void>((resolve, reject) => {
    const pollInterval = setInterval(async () => {
      try {
        // Fetch the current state of the interaction
        // Note: getInteraction returns a BaseMessage. Status is in metadata.
        const message = await agent.getInteraction(interactionId);
        // @ts-ignore
        const status = message.response_metadata.finish_reason as string;

        console.log(`   Current Status: ${status}`);

        // Check for completion states
        // Status values are typically: 'IN_PROGRESS', 'COMPLETED', 'FAILED', 'CANCELLED'
        if (status === 'COMPLETED' || status === 'completed') {
          clearInterval(pollInterval);
          console.log('\n--- Final Report ---');
          console.log(message.content);
          resolve();
        } else if (['FAILED', 'failed', 'CANCELLED', 'cancelled'].includes(status)) {
          clearInterval(pollInterval);
          console.error(`   Job failed: ${status}`);
          resolve();
        }
      } catch (error) {
        console.error('   Polling error:', error);
        clearInterval(pollInterval);
        reject(error);
      }
    }, 5000); // Check every 5 seconds
  });
}

/**
 * 9. Gemini 3 Agent with Tools
 */
async function testGemini3Agent() {
  console.log('\n\n==================================================');
  console.log('RUNNING SUITE: Gemini 3 Agent with Tools');
  console.log('==================================================');

  const model = new ChatGoogleGenAI({
    model: 'gemini-3-flash-preview',
    temperature: 0,
    useExperimentalInteractionsApi: true,
  });

  const agent = createAgent({
    model: model,
    tools: [new Calculator()],
  });

  console.log('User: Calculate 1 + 1 using calculator tool.');
  const res = await agent.invoke({
    messages: [['human', 'Calculate 1 + 1 using calculator tool.']],
  });

  console.log('Agent Response:', JSON.stringify(res.messages, null, 2));
}

/**
 * 10. Store Parameter Test
 */
async function testStoreParameter() {
  console.log('\n\n==================================================');
  console.log('RUNNING SUITE: Interactions API: Store Parameter');
  console.log('==================================================\n');

  const modelNoStore = new ChatGoogleGenAI({
    model: 'gemini-2.5-flash',
    useExperimentalInteractionsApi: true,
    store: false,
  });
  console.log('--- Testing store=false ---');
  const resNoStore = await modelNoStore.invoke('Hello, are you stored?');
  console.log(`[Invoke] AI: ${resNoStore.content}`);
  const interactionId = resNoStore.response_metadata?.interaction_id;
  if (!interactionId) {
    console.log('SUCCESS: Interaction ID is missing as expected.');
  } else {
    console.log('FAILURE: Interaction ID is present despite store=false.');
  }
}

try {
  await testStandardBasic();
  await testInteractionsBasic();
  await testStandardTools();
  await testInteractionsTools();
  await testGemini3Thoughts();
  await testGemini25Thoughts();
  await testStructuredOutput();
  await testStoreParameter();
  await testStoreParameter();
  await testGemini3Agent();
  // await testDeepResearchAgent();
} catch (e) {
  console.error('Test Suite Failed:', e);
}
