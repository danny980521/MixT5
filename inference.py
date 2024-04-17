import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

t5_path = "/data/dannykm/repos/SWC/software_capstone/models/t5_first/checkpoint-21900"
model_paths = [t5_path]
model_names = ["t5"]

for model_name, model_path in zip(model_names, model_paths):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device="cuda")
    print(model)
    _ = model.eval()

    prompts = [
        "[squad][question] what is A? [context] A is apple. B is banana. C is car.",
        "[squad][question] Where was French withdrawal to? [context] Johnson's expedition was better organized than Shirley's, which was noticed by New France's governor, the Marquis de Vaudreuil. He had primarily been concerned about the extended supply line to the forts on the Ohio, and had sent Baron Dieskau to lead the defenses at Frontenac against Shirley's expected attack. When Johnson was seen as the larger threat, Vaudreuil sent Dieskau to Fort St. Frédéric to meet that threat. Dieskau planned to attack the British encampment at Fort Edward at the upper end of navigation on the Hudson River, but Johnson had strongly fortified it, and Dieskau's Indian support was reluctant to attack. The two forces finally met in the bloody Battle of Lake George between Fort Edward and Fort William Henry. The battle ended inconclusively, with both sides withdrawing from the field. Johnson's advance stopped at Fort William Henry, and the French withdrew to Ticonderoga Point, where they began the construction of Fort Carillon (later renamed Fort Ticonderoga after British capture in 1759)."
    ]
    print(f"\n{model_name.upper()} Results: ")
    with torch.no_grad():
        for prompt in prompts:
            tokens = tokenizer.encode(prompt, return_tensors="pt").to(
                device="cuda", non_blocking=True
            )
            gen_tokens = model.generate(tokens, max_length=256, repetition_penalty=1.5)
            generated = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
            print(f"\nQ: {prompt}\nA: {generated}")
