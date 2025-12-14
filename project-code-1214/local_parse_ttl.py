# local_parse_ttl.py
import json
import os

# [경로 확인 필수]
TTL_FILE = "/Users/sunahmin/Desktop/SUN_POSTECH/Courses 수업/2025-2학기/딥러닝구현/Project/code/project-code-1214/artgraph-facts.ttl"
DB_FILE = "artgraph_db.json"

# ID 매핑 (데모용)
ID_MAP = {
    # Artists
    "2016": "Claude Monet", 
    "138": "Vincent van Gogh", "1724": "Vincent van Gogh",
    "60": "Pierre-Auguste Renoir", 
    "2392": "Tsukioka Yoshitoshi",
    "1192": "Henry Scott Tuke", 
    "777": "Fernando Botero",
    "1035": "Paul Cezanne",
    
    # Styles / Movements
    "42": "Impressionism", 
    "21": "Expressionism", 
    "45": "Post-Impressionism",
    "49": "Still Life", 
    "39": "Realism",
    "20": "Baroque",
    "33": "Abstract Art",
    
    # Materials
    "2551": "Oil Paint", 
    "2552": "Canvas",
    
    # Subjects
    "2975": "Water Lilies", "3055": "Pond", "3056": "Reflection",
    "3093": "Nature", "3402": "Summer", "80362": "Sunflowers"
}

def resolve_name(entity_id):
    clean_id = entity_id.replace("artgraph-res:", "").strip()
    return ID_MAP.get(clean_id, clean_id)

def parse_artgraph():
    print(f"Parsing TTL structure from: {TTL_FILE}")
    temp_data = {}
    
    try:
        with open(TTL_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(('@', '#')): continue
                parts = line.replace(" .", "").replace(";", "").split()
                if len(parts) < 3: continue
                
                s_id = parts[0].split(':')[-1]
                p_key = parts[1].split(':')[-1]
                o_raw = " ".join(parts[2:]).replace('"', '')
                
                if s_id not in temp_data:
                    temp_data[s_id] = {
                        'image_name': None, 
                        'facts': [],
                        # [New] 추천 알고리즘을 위한 구조화된 메타데이터
                        'metadata': {'artist': [], 'style': [], 'genre': [], 'material': []}
                    }
                
                # 이미지 매핑
                if 'name' in p_key or 'image_url' in p_key:
                    if o_raw.endswith(('.jpg', '.jpeg', '.png')):
                        temp_data[s_id]['image_name'] = o_raw.split('/')[-1]
                
                # 메타데이터 추출 (Ontology 기반)
                val_clean = o_raw.split(':')[-1].strip()
                if "createdBy" in p_key:
                    temp_data[s_id]['metadata']['artist'].append(val_clean)
                elif "hasStyle" in p_key:
                    temp_data[s_id]['metadata']['style'].append(val_clean)
                elif "hasGenre" in p_key:
                    temp_data[s_id]['metadata']['genre'].append(val_clean)
                elif "madeOf" in p_key:
                    temp_data[s_id]['metadata']['material'].append(val_clean)
                
                # Context용 텍스트 저장
                temp_data[s_id]['facts'].append(f"{p_key}: {o_raw}")

    except FileNotFoundError:
        print("Error: TTL File Not Found!")
        return

    final_db = {}
    for _, info in temp_data.items():
        img_key = info['image_name']
        if img_key:
            clean_key = img_key.split('.')[0]
            
            # 자연어 Context 생성
            meta = info['metadata']
            artist_names = [resolve_name(a) for a in meta['artist']]
            style_names = [resolve_name(s) for s in meta['style']]
            
            desc = f"Title: {img_key}. "
            if artist_names: desc += f"Created by {', '.join(artist_names)}. "
            if style_names: desc += f"Style: {', '.join(style_names)}. "
            
            final_db[clean_key] = {
                "title": img_key,
                "context_text": desc, # LLM에게 줄 문장
                "metadata": info['metadata'] # Python이 쓸 데이터
            }

    with open(DB_FILE, "w", encoding='utf-8') as f:
        json.dump(final_db, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Re-built DB with metadata: {len(final_db)} items.")

if __name__ == "__main__":
    parse_artgraph()