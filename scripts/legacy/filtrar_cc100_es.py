input_path = "data/lm/es.txt"
output_path = "data/lm/cc100_es_clean.txt"

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if 15 < len(line) < 300 and any(p in line for p in ".?!"):
            line = line.replace("\n", " ").replace("\t", " ")
            fout.write(line + "\n")
            
