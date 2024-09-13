from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from secret_key import groq_api_key

import os
os.environ["groq_api_key"] = groq_api_key

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-70b-versatile",
)

def generate_restaurant_info(cuisine):
    # Chain 1: Restaurant Name
    prompt_template_name = PromptTemplate(
        input_variables=["cuisine"],
        template="I want to open a restaurant for {cuisine} food. Suggest a single fancy name for this restaurant. Provide only the name, without any explanation."
    )
    
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")
    
    # Chain 2: Menu Items
    prompt_template_item = PromptTemplate(
        input_variables=["restaurant_name"],
        template="For a restaurant named '{restaurant_name}', suggest 5 popular menu items. Return them as a comma-separated list without numbering or explanation."
    )
    
    food_item_chain = LLMChain(llm=llm, prompt=prompt_template_item, output_key="menu_items")

    chain = SequentialChain(
        chains=[name_chain, food_item_chain],
        input_variables=["cuisine"],
        output_variables=["restaurant_name", "menu_items"]
    )

    response = chain({"cuisine": cuisine})
    
    # Add cuisine to the response dictionary
    response["cuisine"] = cuisine
    
    # Print the output
    print(f"Cuisine: {response['cuisine']}")
    print(f"Restaurant Name: {response['restaurant_name']}")
    print(f"Menu Items: {response['menu_items']}")
    
    return response

if __name__== "__main__":
    print(generate_restaurant_info("Italian"))