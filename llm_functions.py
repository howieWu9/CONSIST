from core.prompt_list import *
import re
from utils.neo4j_operator import deduplicates_list
from configs import TOKEN_COUNT


def _record_tokens(counter):
    TOKEN_COUNT['input'] += int(counter.get('input_token', 0) or 0)
    TOKEN_COUNT['output'] += int(counter.get('output_token', 0) or 0)


def remove_think_tags(text):
    #  <think> 
    pattern = r'<think>.*?</think>'
    #  re.DOTALL  . 
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text

def _parse_llm_response_for_triples_extraction(response_text):
    """
    

    :
        response_text (str): 

    :
        tuple: (entities, relations) 

    :
        1.[entity | ID | Type | "Exact Mention" | Contextual Description]
        2.[relation | SourceID | RelationType | TargetID | "EvidenceSpan"]
    """
    try:
        entities = []
        relations = []
        entity_ids = set()
        entity_name_map = {}

        response_text = remove_think_tags(response_text)
        # 
        lines = [line.strip() for line in response_text.split("\n") if line.strip()]

        for line in lines:
            #  "1."
            modified_line = re.sub(r'^\d+\.', '', line)
            # 
            clean_line = modified_line.replace("[", "", 1).replace("]", "", 1).strip()
            elements = [e.strip() for e in clean_line.split("| ")]

            if not elements:
                continue

            if elements[0] == "entity":
                if len(elements) != 5:
                    raise ValueError(f": {line}")

                entity = {
                    "id": elements[1],
                    "type": elements[2],
                    "mention": elements[3].strip('"'),
                    "description": elements[4]
                }
                entities.append(entity)
                entity_ids.add(entity["id"])
                entity_name_map[entity["id"]] = entity["mention"]

            elif elements[0] == "relation":
                if len(elements) != 5:
                    raise ValueError(f"{line} -> Format Error. Should be [relation | SubjectEntityID | Predicate | ObjectEntityID | \"Evidence or supporting phrase\"].")

                source_id, rel_type, target_id, evidence = elements[1:5]

                if source_id not in entity_ids:
                    raise ValueError(f"SubjectEntityID '{source_id}' not in Entities")
                if target_id not in entity_ids:
                    raise ValueError(f"ObjectEntityID '{target_id}' not in Entities")

                relations.append({
                    "source": source_id,
                    "target": target_id,
                    "relation": rel_type,
                    "evidence": evidence.strip('"'),
                    "source_name": entity_name_map[source_id],
                    "target_name": entity_name_map[target_id]
                })

        if len(entity_ids) != len(entities):
            raise ValueError("ID")

        return entities, relations
    except Exception as e:
        raise e

def extract_triples_from_doc(llm, text):
    prompt = prompt_for_KG_contruction_auto_type(text)
    _token = dict()
    response = llm.call_llm(prompt, post_process=_parse_llm_response_for_triples_extraction, token_counter=_token)
    TOKEN_COUNT['input'] += _token['input_token']
    TOKEN_COUNT['output'] += _token['output_token']
    return response

def _format_context(paths, use_path=True, use_text=True):
    # 
    triples = [
        f"<{triple['begin']['mention']} | {triple['r'].replace('_', ' ')} | {triple['end']['mention']}>"
        for path in paths for triple in path.get('relations', [])
    ]
    format_triples = list(dict.fromkeys(triples))

    # 
    related_sents = [
        sent for path in paths for sent in path.get('context_sentences', [])
    ]
    related_sents = list(dict.fromkeys(related_sents))

    # 
    parts = []
    if use_path and format_triples:
        parts.append("Triples:\n" + "\n".join(format_triples))
    if use_text and related_sents:
        parts.append("Related Text:\n" + "\n".join(related_sents))
    context_info = "\n\n".join(parts)
    return context_info

def _parse_llm_response_for_topic_entity_extraction(response_text):
    """
    Parameters:
        response_text (str): The original response text.

    Returns:
        tuple: (entities, relations) A list of entities and a list of relations.

    Example input:
        1.[entity | ID | Type | "Exact Mention" | Contextual Description]
        2.[relation | SourceID | RelationType | TargetID | "EvidenceSpan"]
    """
    entities = []
    relations = []
    entity_ids = set()
    seen_entity_ids = set()
    entity_name_map = {}

    lines = [line.strip() for line in response_text.split("\n") if line.strip()]

    for line in lines:
        # Remove the numerical index at the beginning of the line(e.g.,"1.").
        modified_line = re.sub(r'^\d+\.', '', line)
        # Remove square brackets and clean up the format
        clean_line = modified_line.replace("[", "", 1).replace("]", "", 1).strip()
        elements = [e.strip() for e in clean_line.split("|")]

        if not elements:
            continue

        if elements[0] == "entity":
            if len(elements) != 5:
                raise ValueError(f"The entity format is incorrect:{line}")

            entity = {
                "id": elements[1],
                "type": elements[2],
                "mention": elements[3].strip('"'),
                "description": elements[4]
            }
            if entity["id"] in seen_entity_ids:
                continue
            entities.append(entity)
            seen_entity_ids.add(entity["id"])
            entity_ids.add(entity["id"])
            entity_name_map[entity["id"]] = entity["mention"]

        elif elements[0] == "relation":
            if len(elements) != 5:
                raise ValueError(f"The relationship format is incorrect:{line}")

            source_id, rel_type, target_id, evidence = elements[1:5]

            if source_id not in entity_ids:
                raise ValueError(f"The source entity ID'{source_id}'does not exist.")
            if target_id not in entity_ids:
                raise ValueError(f"The target entity ID'{target_id}'does not exist.")

            relations.append({
                "source": source_id,
                "target": target_id,
                "relation": rel_type,
                "evidence": evidence.strip('"'),
                "source_name": entity_name_map[source_id],
                "target_name": entity_name_map[target_id]
            })

    return entities

def topic_entity_extraction(llm, text):
    # prompt = prompt_for_entity_extraction(text)
    prompt = prompt_for_preprocess_query(text)
    _token = {'input_token': 0, 'output_token': 0}
    entities = llm.call_llm(prompt, post_process=_parse_llm_response_for_topic_entity_extraction, token_counter=_token)
    TOKEN_COUNT['input'] += _token['input_token']
    TOKEN_COUNT['output'] += _token['output_token']
    return entities

def _parse_llm_response_for_score_triples(response_text, **kwargs):
    try:
        candidates = kwargs.get('candidates', [])

        #  0, 0.0, 0.25, 1, 1.0
        pattern = re.compile(
            r'<([^>]+)>\s*:\s*(\d+(?:\.\d+)?)'
        )

        results = []
        for match in pattern.finditer(response_text):
            triplet_part, score_str = match.groups()

            #  split LLM  ValueError
            parts = [p.strip() for p in triplet_part.split('|')]
            if len(parts) != 3:
                continue  #  raise ValueError("Invalid triple format")

            head, relation, tail = parts
            score = float(score_str)

            results.append({
                'head': head,
                'relation': relation,
                'tail': tail,
                'score': score
            })

        # 
        if len(results) == len(candidates):
            return results

        #  candidates 
        outputs = [
            {
                'head': candidate['begin']['mention'],
                'relation': candidate['r'],
                'tail': candidate['end']['mention'],
                'score': 0.0
            }
            for candidate in candidates
        ]

        # 
        tri2scores = {
            (r['head'], r['relation'], r['tail']): r['score']
            for r in results
        }

        #  0.0 
        for tri in outputs:
            key = (tri['head'], tri['relation'], tri['tail'])
            if key in tri2scores:
                tri['score'] = tri2scores[key]

        return outputs

    except Exception as e:
        raise e

def score_triples(llm, query, paths, batch_size=20):
    all_results = []
    # Process candidate triples in batches
    for i in range(0, len(paths), batch_size):
        batch_path = paths[i:i + batch_size]
        container = [r for p in batch_path for r in p.relations[:-1]]
        candidates = [p.relations[-1] for p in batch_path]
        batch_results = _score_triples_in_batch(llm, query, container, candidates)
        all_results.extend(batch_results)
    assert len(all_results) == len(paths)
    return all_results

def _score_triples_in_batch(llm, query, container, candidates):
    format_container = [f"<{triple['begin']['mention']} | {triple['r']} | {triple['end']['mention']}>" for triple in
                        container]
    format_candidates = [f"<{triple['begin']['mention']} | {triple['r']} | {triple['end']['mention']}>" for triple in
                         candidates]
    existed_text = "\n".join(deduplicates_list(format_container)) if len(format_container) else "None"
    text = "\n".join(format_candidates)
    prompt = prompt_for_score_triples(query, text, existed_text)

    _token = {'input_token': 0, 'output_token': 0}
    results = llm.call_llm(prompt, post_process=_parse_llm_response_for_score_triples, candidates=candidates,
                           token_counter=_token)
    TOKEN_COUNT['input'] += _token['input_token']
    TOKEN_COUNT['output'] += _token['output_token']

    return results

def _parse_llm_response_for_eval_sufficiency_with_llm(response_text, **kwargs):
    if 'yes' in response_text.lower():
        return True
    elif 'no' in response_text.lower():
        return False
    else:
        raise ValueError("The answer format is incorrect!")

def eval_sufficiency_with_llm(llm, paths, query):
    context_info = _format_context(paths)
    prompt = prompt_for_eval_sufficiency(context_info, query)
    _token = {'input_token': 0, 'output_token': 0}
    results = llm.call_llm(prompt, post_process=_parse_llm_response_for_eval_sufficiency_with_llm, token_counter=_token)
    TOKEN_COUNT['input'] += _token['input_token']
    TOKEN_COUNT['output'] += _token['output_token']

    return results


def _parse_llm_response_for_reasoning(response_text, **kwargs):
    """
     Final Answer

    :
    output_text (str): 

    :
    str: 
    """
    # /
    pattern = r"(?i)(?:[\*\_]+)?Final Answer(?:[\*\_]+)?\s*:\s*(.*)"
    match = re.search(pattern, response_text, re.DOTALL)

    if match:
        answer = match.group(1).strip()
        # 
        return ' '.join(answer.split()).strip()
    raise ValueError(f" Final Answer: \n{response_text}")


def reasoning(llm, question, text):
    prompt = prompt_for_reasoning_tcr(question, text)
    # res = llm.call_llm(prompt, post_process=_parse_llm_response_for_reasoning)

    _token = {'input_token': 0, 'output_token': 0}
    res = llm.call_llm(prompt, post_process=_parse_llm_response_for_reasoning, token_counter=_token)
    TOKEN_COUNT['input'] += _token['input_token']
    TOKEN_COUNT['output'] += _token['output_token']
    return res


def reasoning_RAG(llm, question, text):
    prompt = prompt_for_RAG_llm(text, question)
    # res = llm.call_llm(prompt, post_process=_parse_llm_response_for_reasoning)

    _token = {'input_token': 0, 'output_token': 0}
    res = llm.call_llm(prompt, token_counter=_token)
    TOKEN_COUNT['input'] += _token['input_token']
    TOKEN_COUNT['output'] += _token['output_token']
    return res

def reasoning_LLM_only(llm, question):
    prompt = prompt_for_llm_only(question)
    # res = llm.call_llm(prompt, post_process=_parse_llm_response_for_reasoning)
    _token = {'input_token': 0, 'output_token': 0}
    res = llm.call_llm(prompt, token_counter=_token)
    TOKEN_COUNT['input'] += _token['input_token']
    TOKEN_COUNT['output'] += _token['output_token']
    return res

def _parse_llm_response_for_missing_knowledge_identify(response_text, **kwargs):
    """
     Final Answer

    :
    output_text (str): 

    :
    str: 
    """
    # /
    questions = response_text.strip().split('\n')
    assert len(questions) > 0
    return questions

def missing_knowledge_identify(llm, question, path):
    context_info = _format_context(path, use_text=False)
    prompt = prompt_for_missing_knowledge_identify(context_info, question)
    _token = {'input_token': 0, 'output_token': 0}
    sub_questions = llm.call_llm(
        prompt,
        post_process=_parse_llm_response_for_missing_knowledge_identify,
        token_counter=_token,
    )
    _record_tokens(_token)
    return sub_questions

def missing_knowledge_extraction(llm, question, path, sub_questions, context, **kwargs):
    try:
        context_info = _format_context(path, use_text=False)
        prompt = prompt_for_missing_knowledge_extraction_refine(context_info, sub_questions, context)
        _token = {'input_token': 0, 'output_token': 0}
        entities, relations = llm.call_llm(
            prompt,
            post_process=_parse_llm_response_for_triples_extraction,
            token_counter=_token,
        )
        _record_tokens(_token)
        return entities, relations
    except Exception as e:
        raise e


def _parse_llm_response_for_multi_relation(response_text: str, **kwargs) -> list:
    """
    
    '|'
    """
    relations_data = []
    lines = response_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or ':' not in line:
            continue
        try:
            pair_str, relation_str = line.rsplit(':', 1)
            #  '||'  entity_pairs_str 
            pair_str, relation_str = line.rsplit(':', 1)
            if "||" in pair_str:
                head, tail = [x.strip() for x in pair_str.split('||', 1)]
            elif "|" in pair_str:
                head, tail = [x.strip() for x in pair_str.split('|', 1)]
            else:
                continue

            relation_str = relation_str.strip()

            if 'none' in relation_str.lower():
                relations_list = []
            else:
                relations_list = [rel.strip() for rel in relation_str.split('|') if rel.strip()]

            relations_data.append({
                'head': head,
                'tail': tail,
                'relations': relations_list
            })
        except Exception:
            continue

    return relations_data

def complete_relations(llm, text, entity_pairs, query):
    entity_pairs_str = ""
    for entity_pair in entity_pairs:
        entity_pairs_str += f"{entity_pair[0].strip()} || {entity_pair[1].strip()}\n"
    prompt = prompt_for_focused_multi_relation_completion(text, entity_pairs_str, query)
    _token = {'input_token': 0, 'output_token': 0}
    result = llm.call_llm(
        prompt,
        post_process=_parse_llm_response_for_multi_relation,
        request_overrides={'temperature': 0},
        token_counter=_token,
    )
    _record_tokens(_token)
    return result

def summary_answer(llm, text, question):
    prompt = prompt_for_summary_answer(text, question)
    _token = {'input_token': 0, 'output_token': 0}
    result = llm.call_llm(prompt, token_counter=_token)
    _record_tokens(_token)
    return result
