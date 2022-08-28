import transformers
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
generator(
    "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone"
)  # doctest: +SKIP

# generator(
#     [
#         "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone",
#         "Nine for Mortal Men, doomed to die, One for the Dark Lord on his dark throne",
#     ]
# )  # doctest: +SKIP

# generator(
#     "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone",
#     num_return_sequences=2,
# )  # doctest: +SKIP

