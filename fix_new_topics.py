import json
from pathlib import Path

notebooks = [
    "notebooks/models/topicGpt/continuity/continuity_rate.ipynb",
    "notebooks/models/lda/continuity/continuity_rate.ipynb",
    "notebooks/models/bertopic/continuity/continuity_rate.ipynb",
    "notebooks/models/dtm/continuity/continuity_rate.ipynb"
]

target_lines = [
    "        # ── Step 5: New topics — no incoming source with sim > 0 ────────────\n",
    "        matched_t1 = set(topics_t1[best_match_idx[idx]]\n",
    "                         for idx in range(len(topics_t))\n",
    "                         if float(best_match_sim[idx]) > 0)\n",
    "        new_topics = [tid for tid in topics_t1 if tid not in matched_t1]\n"
]

replacement_lines = [
    "        # ── Step 5: New topics — no incoming source with sim > 0 ────────────\n",
    "        new_topics = []\n",
    "        for jj, tid in enumerate(topics_t1):\n",
    "            if np.max(sim_matrix[:, jj]) == 0:\n",
    "                new_topics.append(tid)\n"
]

for nb_path in notebooks:
    p = Path(nb_path)
    if not p.exists():
        print(f"Skipping {nb_path}, does not exist")
        continue
        
    with open(p, "r") as f:
        data = json.load(f)
        
    modified = False
    for cell in data.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            # Find the index of the first target line
            for i in range(len(source) - len(target_lines) + 1):
                if source[i:i+len(target_lines)] == target_lines:
                    # Replace
                    source[i:i+len(target_lines)] = replacement_lines
                    cell["source"] = source
                    modified = True
                    break
                    
    if modified:
        with open(p, "w") as f:
            json.dump(data, f, indent=1)
            # Ensure proper trailing newline and spaces
            f.write("\n")
        print(f"Modified {nb_path}")
    else:
        print(f"No match found in {nb_path}")

