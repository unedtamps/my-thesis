import json
import sys

def update_notebook(path):
    with open(path, 'r') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
            
        src = "".join(cell.get('source', []))
        
        # 0. remove imports
        if "from sklearn.metrics.pairwise import cosine_similarity\n" in src:
            src = src.replace("from sklearn.metrics.pairwise import cosine_similarity\n", "")
        
        # 1. Update helpers
        old_helper = (
            "def words_to_vector(words, vocab_index):\n"
            "    vec = np.zeros(len(vocab_index))\n"
            "    for w in words:\n"
            "        if w in vocab_index:\n"
            "            vec[vocab_index[w]] = 1.0\n"
            "    return vec\n\n\n"
            "def cos_sim(v1, v2):\n"
            "    \"\"\"Cosine similarity between two vectors.\"\"\"\n"
            "    return float(cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0])"
        )
        new_helper = (
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
        if old_helper in src:
            src = src.replace(old_helper, new_helper)
            
        # 2. Endpoint TTD
        old_end_vocab = (
            "    # Build global vocabulary\n"
            "    all_w = set()\n"
            "    for ws in topic_words.values():\n"
            "        all_w.update(ws)\n"
            "    vocab_index = {w: i for i, w in enumerate(sorted(all_w))}\n"
        )
        src = src.replace(old_end_vocab, "")
        
        old_end_sim = (
            "        v_first = words_to_vector(topic_words[(t_first, tid)], vocab_index)\n"
            "        v_last = words_to_vector(topic_words[(t_last, tid)], vocab_index)\n"
            "        sim = cos_sim(v_first, v_last)"
        )
        new_end_sim = (
            "        sim = rbo(topic_words[(t_first, tid)], topic_words[(t_last, tid)], p=0.9)"
        )
        src = src.replace(old_end_sim, new_end_sim)
        
        # 3. Trajectory TTD
        old_traj_vocab = (
            "    all_w = set()\n"
            "    for ws in topic_words.values():\n"
            "        all_w.update(ws)\n"
            "    vocab_index = {w: i for i, w in enumerate(sorted(all_w))}\n"
        )
        src = src.replace(old_traj_vocab, "")
        
        old_traj_base = "        v_baseline = words_to_vector(topic_words[(t0, tid)], vocab_index)"
        new_traj_base = "        words_baseline = topic_words[(t0, tid)]"
        src = src.replace(old_traj_base, new_traj_base)
        
        old_traj_sim = (
            "            v_y = words_to_vector(topic_words[(y, tid)], vocab_index)\n"
            "            sim = cos_sim(v_baseline, v_y)"
        )
        new_traj_sim = (
            "            words_y = topic_words[(y, tid)]\n"
            "            sim = rbo(words_baseline, words_y, p=0.9)"
        )
        src = src.replace(old_traj_sim, new_traj_sim)

        # 4. YoY TTD
        old_yoy_vocab = (
            "    # Build global vocabulary\n"
            "    all_w = set()\n"
            "    for ws in topic_words.values():\n"
            "        all_w.update(ws)\n"
            "    vocab_index = {w: i for i, w in enumerate(sorted(all_w))}\n"
        )
        src = src.replace(old_yoy_vocab, "")
        
        old_yoy_sim = (
            "            v_t = words_to_vector(topic_words[(t, tid)], vocab_index)\n"
            "            v_t1 = words_to_vector(topic_words[(t1, tid)], vocab_index)\n"
            "            sim = cos_sim(v_t, v_t1)"
        )
        new_yoy_sim = (
            "            words_t = topic_words[(t, tid)]\n"
            "            words_t1 = topic_words[(t1, tid)]\n"
            "            sim = rbo(words_t, words_t1, p=0.9)"
        )
        src = src.replace(old_yoy_sim, new_yoy_sim)
        
        # 5. Inspect Functions - replace the vocab logic correctly
        old_inspect1_vocab = (
            "    all_words_set = set()\n"
            "    tw = {}\n"
            "    for _, row in topic_evo.iterrows():\n"
            "        words = parse_words(row[\"top_words\"])\n"
            "        tw[int(row[\"year\"])] = words\n"
            "        all_words_set.update(words)\n"
            "    vi = {w: i for i, w in enumerate(sorted(all_words_set))}\n"
        )
        new_inspect1 = (
            "    tw = {}\n"
            "    for _, row in topic_evo.iterrows():\n"
            "        tw[int(row[\"year\"])] = parse_words(row[\"top_words\"])\n"
        )
        src = src.replace(old_inspect1_vocab, new_inspect1)
        
        old_inspect2_vocab = (
            "    tw = {}\n"
            "    all_words_set = set()\n"
            "    for _, row in topic_evo.iterrows():\n"
            "        words = parse_words(row[\"top_words\"])\n"
            "        tw[int(row[\"year\"])] = words\n"
            "        all_words_set.update(words)\n"
            "    vi = {w: i for i, w in enumerate(sorted(all_words_set))}\n"
        )
        new_inspect2 = (
            "    tw = {}\n"
            "    for _, row in topic_evo.iterrows():\n"
            "        tw[int(row[\"year\"])] = parse_words(row[\"top_words\"])\n"
        )
        src = src.replace(old_inspect2_vocab, new_inspect2)

        # Replacing inspect logic
        old_yoy_inspect_sim = (
            "        v_t = words_to_vector(tw[t], vi)\n"
            "        v_t1 = words_to_vector(tw[t1], vi)\n"
            "        sim = cos_sim(v_t, v_t1)"
        )
        new_yoy_inspect_sim = "        sim = rbo(tw[t], tw[t1], p=0.9)"
        src = src.replace(old_yoy_inspect_sim, new_yoy_inspect_sim)
        
        old_end_inspect_sim = (
            "    v_first = words_to_vector(tw[t_first], vi)\n"
            "    v_last = words_to_vector(tw[t_last], vi)\n"
            "    sim = cos_sim(v_first, v_last)"
        )
        new_end_inspect_sim = "    sim = rbo(tw[t_first], tw[t_last], p=0.9)"
        src = src.replace(old_end_inspect_sim, new_end_inspect_sim)

        old_traj_inspect_base = "    v_baseline = words_to_vector(tw[t0], vi)"
        new_traj_inspect_base = "    words_baseline = tw[t0]"
        src = src.replace(old_traj_inspect_base, new_traj_inspect_base)
        
        old_traj_inspect_loop = (
            "        v_y = words_to_vector(tw[y], vi)\n"
            "        sim = cos_sim(v_baseline, v_y)"
        )
        new_traj_inspect_loop = (
            "        words_y = tw[y]\n"
            "        sim = rbo(words_baseline, words_y, p=0.9)"
        )
        src = src.replace(old_traj_inspect_loop, new_traj_inspect_loop)

        old_traj_inspect_final = "    final_sim = cos_sim(v_baseline, words_to_vector(tw[topic_years[-1]], vi))"
        new_traj_inspect_final = "    final_sim = rbo(words_baseline, tw[topic_years[-1]], p=0.9)"
        src = src.replace(old_traj_inspect_final, new_traj_inspect_final)

        cell['source'] = src.splitlines(True)
        
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    for p in sys.argv[1:]:
        update_notebook(p)
