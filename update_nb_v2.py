import json
import sys

def modify_notebook(path):
    with open(path, 'r') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell.get('source', [])
            src_str = "".join(source)
            
            # Replace import
            if "from sklearn.metrics.pairwise import cosine_similarity\n" in src_str:
                src_str = src_str.replace(
                    "from sklearn.metrics.pairwise import cosine_similarity\n",
                    ""
                )
            
            # Replace helper function
            old_func = (
                "def build_word_matrix(topic_ids, topic_words_dict, year, vocab_index):\n"
                "    matrix = np.zeros((len(topic_ids), len(vocab_index)))\n"
                "    for i, tid in enumerate(topic_ids):\n"
                "        words = topic_words_dict.get((year, tid), [])\n"
                "        for w in words:\n"
                "            if w in vocab_index:\n"
                "                matrix[i, vocab_index[w]] = 1.0\n"
                "    return matrix"
            )
            new_func = (
                "def rbo(list_1, list_2, p=0.9):\n"
                "    k = min(len(list_1), len(list_2))\n"
                "    if k == 0:\n"
                "        return 0.0\n"
                "    score = 0.0\n"
                "    for d in range(1, k + 1):\n"
                "        agreement = len(set(list_1[:d]) & set(list_2[:d])) / d\n"
                "        score += (p ** (d - 1)) * agreement\n"
                "    return score * (1 - p)"
            )
            if old_func in src_str:
                src_str = src_str.replace(old_func, new_func)
                
            # Replace vocab loop
            old_vocab = (
                "    all_words = set()\n"
                "    for words in topic_words.values():\n"
                "        all_words.update(words)\n"
                "    vocab_index = {w: i for i, w in enumerate(sorted(all_words))}\n\n"
            )
            if old_vocab in src_str:
                src_str = src_str.replace(old_vocab, "")
                
            # Replace sim_matrix calculation
            old_sim = (
                "        mat_t = build_word_matrix(topics_t, topic_words, t, vocab_index)\n"
                "        mat_t1 = build_word_matrix(topics_t1, topic_words, t_next, vocab_index)\n"
                "        sim_matrix = cosine_similarity(mat_t, mat_t1)"
            )
            new_sim = (
                "        sim_matrix = np.zeros((len(topics_t), len(topics_t1)))\n"
                "        for i, tid_t in enumerate(topics_t):\n"
                "            words_t = topic_words.get((t, tid_t), [])\n"
                "            for j, tid_t1 in enumerate(topics_t1):\n"
                "                words_t1 = topic_words.get((t_next, tid_t1), [])\n"
                "                sim_matrix[i, j] = rbo(words_t, words_t1, p=0.9)"
            )
            if old_sim in src_str:
                src_str = src_str.replace(old_sim, new_sim)

            lines = src_str.splitlines(True)
            cell['source'] = lines

    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    modify_notebook(sys.argv[1])
