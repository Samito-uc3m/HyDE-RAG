import os
os.environ['OLLAMA_HOST'] = "http://kumo01:11434"

import ollama
response = ollama.chat(
	model="llama3.2",
	messages=[
		{
			'role': 'user',
			'content': 'Describe me what topic modeling is as if I were a kid',
		},
	]
)
print(response['message']['content'])