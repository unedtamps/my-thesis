import json
import sys

def modify_notebook(path):
    with open(path, 'r') as f:
        nb = json.load(f)

    # Modify cells
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = cell.get('source', [])
            new_source = []
            
            skip_lines = False
            for i, line in enumerate(source):
                if line == "from sklearn.metrics.pairwise import cosine_similarity\n":
                    continue # skip this line
                
                # Replace build_word_matrix definition
                if line == "def build_word_matrix(topic_ids, topic_words_dict, year, vocab_index):\n":
                    skip_lines = True
                    new_source.append("def rbo(list_1, list_2, p=0.9):\n")
                    continue
                elif skip_lines:
                    if line == "    return matrix\n" or line == "    return matrix":
                        skip_lines = False
                    continue
                
                # Strip vocab_index creation
                if line == "    all_words = set()\n":
                    skip_lines = True
                    continue
                elif skip_lines and "vocab_index = {w: i for i, w in enumerate(sorted(all_words))}" in line:
                    skip_lines = False
                    continue
                elif skip_lines and ("all_words.update(words)" in line or "for words in topic_words.values():" in line):
                    continue
                
                # Replace loop
                if line == "        mat_t = build_word_matrix(topics_t, topic_words, t, vocab_index)\n":
                    new_source.extend([
                        "        sim_matrix = np.zeros((len(topics_t), len(topics_t1)))\n",
                        "        for i, tid_t in enumerate(topics_t):\n",
                        "            words_t = topic_words.get((t, tid_t), [])\n",
                        "            for j, tid_t1 in enumerate(topics_t1):\n",
                        "                words_t1 = topic_words.get((t_next, tid_t1), [])\n",
                        "                sim_matrix[i, j] = rbo(words_t, words_t1, p=0.9)\n"
                    ])
                    continue
                elif line == "        mat_t1 = build_word_matrix(topics_t1, topic_words, t_next, vocab_index)\n":
                    continue
                elif line == "        sim_matrix = cosine_similarity(mat_t, mat_t1)\n":
                    continue
                    
                new_source.append(line)
            
            # Now we add the rbo function content if we inserted the def
            final_source = []
            for line in new_source:
                final_source.append(line)
                if line == "def rbo(list_1, list_2, p=0.9):\n":
                    final_source.extend([
                        "    k = min(len(list_1), len(list_2))\n",
                        "    if k == 0:\n",
                        "        return 0.0\n",
                        "    score = 0.0\n",
                        "    for d in range(1, k + 1):\n",
                        "        agreement = len(set(list_1[:d]) & set(list_2[:d])) / d\n",
                        "        score += (p ** (d - 1)) * agreement\n",
                        "    return score * (1 - p)\n"
                    ])
            
            cell['source'] = final_source

    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
        # Notebook format often expects indent=1 and no trailing spaces, wait actually standard is indent=1 or indent=2. Let's just write and it's fine.
        # It's better to read and write using python exactly if possible but the json lib works.

if __name__ == "__main__":
    modify_notebook(sys.argv[1])
