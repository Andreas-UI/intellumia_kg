import os
from dotenv import load_dotenv

from doc_chunk import doc_chunk
from helpers_people_windows import persons_in_chunk, windows_for_person
from llm import LLMConfig, OpenAIClient
from entities_mentions import p1_entities_mentions
from coref_canonicalize import p2_coref_canonicalize
from relations_events import p3_relations_events
from trait_window import (
    calibrate_trait_probs,
    convert_trait_json_to_evidence,
    p4_trait_window,
)
from r2_fuse_relations_events import r2_fuse_relations_events
from r3_aggregate_traits_beta import r3_aggregate_traits_beta

import networkx as nx
from pyvis.network import Network

load_dotenv()


def main(file_path):
    # Document into several chunks
    print("1. Chunking document...")
    chunks = doc_chunk(file_path)
    print("1. Chunking document - DONE\n")

    client = OpenAIClient(
        LLMConfig(model="gpt-4o-mini"),
        api_key=os.getenv("OPENAI_KEY"),
    )

    # P1: Entities
    print("2. Getting all entities...")
    ents_results = []
    chunk_texts = {}
    for chunk in chunks:
        ents_results.append(
            p1_entities_mentions(client, chunk["chunk_id"], chunk["text"])
        )
        chunk_texts[chunk["chunk_id"]] = chunk["text"]
    print("2. Getting all entities - DONE\n")

    # P2: Coreference
    print("3. Coreferencing entities...")
    coref = p2_coref_canonicalize(
        client,
        p1_chunks=ents_results,
        chunk_texts=chunk_texts,
        high_sim_threshold=0.93,
        low_sim_threshold=0.40,
    )
    print("3. Coreferencing entities - DONE\n")

    rels_ev_all = []
    trait_evs = []
    chunk_idx = 1
    for chunk, ents in zip(chunks, ents_results):
        print(f"\t4. Extracting event relations ({chunk_idx})...")
        # P3: Relations Events
        rels_ev = p3_relations_events(
            llm=client,
            chunk_id=chunk["chunk_id"],
            chunk_text=chunk["text"],
            entities_in_chunk=ents["entities"],
            coref_map_local_to_global=coref["coref_map"],
        )
        print(f"\t4. Extracting event relations ({chunk_idx}) - DONE\n")

        rels_ev_all.append(rels_ev)

        # --- P4 windows ---
        people = persons_in_chunk(
            ents["entities"], coref["coref_map"], chunk["chunk_id"]
        )

        print(f"\t5. Calculate each person trait ({chunk_idx})...")
        for person in people:
            wins = windows_for_person(
                chunk["text"], chunk["sent_index"], person["mentions"], window_size=1
            )
            for w in wins:
                w["chunk_id"] = chunk["chunk_id"]

                # P4: Trait
                tr = p4_trait_window(
                    client,
                    person_name=person["display_name"],
                    window_text=w["text"],
                    require_spans=True,
                )
                cal = calibrate_trait_probs(tr["trait_probs"])
                evs = convert_trait_json_to_evidence(
                    person_id=person["canonical_id"],
                    window_meta=w,
                    trait_probs=cal,
                    evidence=tr["evidence"],
                )
                trait_evs.extend(evs)
        print(f"\t5. Calculate each person trait ({chunk_idx}) - DONE\n")
        chunk_idx += 1

    print("Finalizing everything...")
    rels_ev_fused = r2_fuse_relations_events(rels_ev_all)
    trait_edges = r3_aggregate_traits_beta(trait_evs, tau_score=0.65, tau_conf=0.60)
    print("Finalizing everything... - DONE")

    return coref, rels_ev_fused, trait_edges


def build_nx_graph(entities_canonical, rels_ev_fused, trait_edges):
    G = nx.MultiDiGraph()

    # 1) Add entity nodes
    for ent in entities_canonical:
        G.add_node(
            ent["id"],
            label=ent["name"],
            group=ent["type"],  # Person/Org/Location/...
            title=f"Aliases: {', '.join(ent.get('aliases', []))}",
        )

    # 2) Add Trait nodes (one per unique trait) — optional separate namespace
    trait_nodes = set(t["trait"] for t in trait_edges)
    for tname in trait_nodes:
        node_id = f"trait::{tname}"
        G.add_node(node_id, label=tname, group="Trait")

    # 3) Add factual relation edges
    for r in rels_ev_fused.get("relations", []):
        if r["head"] in G and r["tail"] in G:
            lbl = r["type"]
            title = f"surface: {r.get('source_label', '')}\nconf: {r.get('confidence_local', 0):.2f}"
            G.add_edge(r["head"], r["tail"], label=lbl, title=title, color="#8899ff")

    # 4) Add trait edges
    for e in trait_edges:
        trait_node = f"trait::{e['trait']}"
        if e["person"] in G and trait_node in G:
            G.add_edge(
                e["person"],
                trait_node,
                label=f"{e['score']:.2f}/{e['confidence']:.2f}",
                title=f"n_evidence: {e['n_evidence']}",
                color="#33aa66",
            )

    return G


def draw_pyvis(G, out_html="kg_graph.html"):
    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.from_nx(G)
    # nicer physics
    net.barnes_hut(
        gravity=-2000, central_gravity=0.2, spring_length=150, spring_strength=0.01
    )
    net.show(out_html, notebook=False)
    return out_html


coref, rels_ev_fused, trait_edges = main("doc.txt")

print("Building Graph")

G = build_nx_graph(
    entities_canonical=coref["entities_canonical"],
    rels_ev_fused=rels_ev_fused,
    trait_edges=trait_edges,
)
draw_pyvis(G, out_html="kg_graph.html")
