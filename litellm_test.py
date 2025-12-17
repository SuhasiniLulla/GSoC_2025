from litellm import completion
import json

## COMPLETION CALL
response = completion(
  model="vertex_ai/gemini-2.5-pro",
  #model="vertex_ai/gemini-3-pro-preview",
  messages=[{ "content": "tell me a joke?", "role": "user"}]
)

print(response)
