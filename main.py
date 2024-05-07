import random
import autogen

config_list = autogen.config_list_from_json(
  "OAI_CONFIG_LIST.json",
  filter_dict={
    "model": ["gpt-3.5-turbo-1106"]
  }
)

llm_config = {
  "temperature": 0,
  "timeout": 300,
  "seed": random.randint(100, 100000),
  "config_list": config_list
}
