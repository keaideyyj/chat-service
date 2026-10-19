from openai import OpenAI

from app.config import get_settings

s = get_settings()
client = OpenAI(api_key=s.deepseek_api_key, base_url=s.deepseek_base_url)
for m in client.models.list().data:
    print(m.id)