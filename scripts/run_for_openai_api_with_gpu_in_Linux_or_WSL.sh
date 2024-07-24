#!/bin/bash
# bash scripts/base_run.sh -s "LinuxOrWSL" -w 4 -m 19530 -q 8777 -o -b 'https://api.openai.com/v1' -k 'sk-xxx' -n 'gpt-3.5-turbo-1106' -l '4096'

bash scripts/base_run.sh -s "LinuxOrWSL" -w 4 -m 19530 -q 8777 -o -b 'http://localhost:11434/v1' -k 'ollama' -n 'qwen2' -l '4096'
