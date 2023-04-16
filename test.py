import os
import openai
openai.api_key = os.getenv("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")
openai.Completion.create(
  model="text-davinci-002",
  prompt="Say this is a test",
  max_tokens=6,
  temperature=0
)