"""
1) Messages: For each of the 20 topics, compiled 28-30 concrete and actionable target behaviors using GPT-4o
2) Persuaion strategies: Five persuasion strategies (first four strategies into P/N frames)
3) Premises: Three premises for each persuasion strategy using GPT-4o
4) Queries: Converted each premise into a suitable prompt for DALLE and a search query for Google using GPT-4o
5) Images: one image from DALLE and on from Google Image Search for each premise (54 images)
6) Persuasive Scores: 0 to 10
7) Personal Information: Annotator ID, Age, Gender, Habit. Psychological Characteristics (Big5, PVQ21, MFQ30)
"""

"""
message: 
"""

import openai

openai.api_key = "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" 

response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
        {"role": "user", "content": "Please create persuasive messages that demand behavioral change, following these conditions:\
         1. They must be universal and not violate common sense.\
         2. They must be immediately relatable and something that an average person can do.\
         3. The topic shoub be about sustainable food choices, food safety, and eco-freindly practices.\
         4. Generate 30 distinct messages that do not overlap with each other.\
         5. Exclude any reasoning; the messages should be direct and action-oriented.\
         Here is an example:\
         1. Purchase organic food.\
         2. Comsume seasonal produce."}
    ]
)

print(response["choices"][0]["message"]["content"])

