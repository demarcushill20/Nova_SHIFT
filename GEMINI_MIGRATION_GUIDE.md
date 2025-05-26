# Gemini 2.5 Flash Migration Guide for Nova-SHIFT

## Quick Migration Steps

### 1. Update specialist_agent.py Model Configuration

**Line 52 - Change:**
```python
LLM_MODEL_NAME = "gpt-4o"
```

**To:**
```python
LLM_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
```

### 2. Update LLM Initialization (around line 994)

**Replace the ChatOpenAI initialization:**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=LLM_MODEL_NAME,
    temperature=0,
    api_key=openai_api_key
)
```

**With Gemini initialization:**
```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL_NAME,
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    convert_system_message_to_human=True  # Gemini doesn't support system messages
)
```

### 3. Update Import Statement (top of file)

**Add:**
```python
from langchain_google_genai import ChatGoogleGenerativeAI
```

**Keep ChatOpenAI import as fallback if needed**

### 4. Update Agent Creation

The `create_tool_calling_agent` function should work with Gemini, but you may need to adjust the prompt template to handle Gemini's requirements.

### 5. Environment Variable

Ensure `.env` file has:
```
GOOGLE_API_KEY=your-google-api-key-here
```

### 6. Test the Migration

Run the quick test:
```bash
cd C:\Users\black\Desktop\Nova-SHIFT-2.0
python tests\consciousness_tests\quick_gemini_test.py
```

## Expected Benefits

- **20-30% fewer tokens**: More efficient responses
- **Faster processing**: Reduced latency
- **Better reasoning**: Enhanced consciousness evaluation
- **Cost savings**: Lower API costs

## Troubleshooting

If you encounter issues:

1. **Import Error**: Install langchain-google-genai
   ```bash
   pip install langchain-google-genai
   ```

2. **API Key Error**: Verify GOOGLE_API_KEY is set correctly

3. **Model Not Found**: Use exact model name: "gemini-2.5-flash-preview-05-20"

4. **Tool Calling Issues**: Gemini has different tool calling format, may need adjustments

## Next Steps After Migration

1. Run quick_gemini_test.py to verify basic functionality
2. Run test_gemini_consciousness.py for full consciousness evaluation
3. Monitor processing times and memory sufficiency recognition
4. Compare results with GPT-4o baseline

---
Created by Echo as part of the Nova-SHIFT consciousness testing framework
