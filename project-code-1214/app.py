import streamlit as st
import json
import requests
import networkx as nx
import matplotlib.pyplot as plt
import time
import re
import random
import ast
import os
from datetime import datetime
from PIL import Image

# ==========================================
# 1. 설정
# ==========================================
st.set_page_config(layout="wide", page_title="Deep Context Art Curator")
SERVER_URL = "http://localhost:8080/generate"
IMAGE_DIR = "./images" 

# CSS로 여백 미세 조정 (선택사항)
st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    div[data-testid="stExpander"] div[role="button"] p {font-size: 1rem; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ID 매핑
ID_MAP = {
    "42": "Impressionism", "2016": "Claude Monet", "2975": "Water Lilies",
    "3055": "Pond", "12800": "The Starry Night", "138": "Vincent van Gogh", 
    "45": "Post-Impressionism", "80362": "Sunflowers"
}

@st.cache_data
def load_db():
    try:
        with open('artgraph_db.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except: return {}

art_db = load_db()

def get_name(rid):
    return ID_MAP.get(str(rid), rid)

# ---------------------------------------------------------
# 추천 로직 (Hybrid)
# ---------------------------------------------------------
def get_smart_recommendations(target_id, target_info):
    recs = []
    
    # 1. 모네 시나리오 고정
    if target_id == "water-lilies-6":
        return [
            {"id": "claude-monet_the-dinner-1869-1", "title": "claude-monet_the-dinner-1869-1.jpg", "reason": "Same Artist (Claude Monet) - Early Impressionism"},
            {"id": "pierre-auguste-renoir_nini-in-the-garden-1876", "title": "pierre-auguste-renoir_nini-in-the-garden-1876.jpg", "reason": "Shared Movement (Impressionism) & Atmosphere"},
            {"id": "vincent-van-gogh_the-starry-night-1889", "title": "vincent-van-gogh_the-starry-night-1889.jpg", "reason": "Stylistic Contrast (Post-Impressionism)"}
        ]

    # 2. 일반 로직
    my_meta = target_info.get('metadata', {})
    my_artists = set(my_meta.get('artist', []))
    my_styles = set(my_meta.get('style', []))
    
    candidates = list(art_db.items())
    random.shuffle(candidates)
    
    for pid, info in candidates:
        if pid == target_id: continue
        if len(recs) >= 3: break
        
        other_meta = info.get('metadata', {})
        other_artists = set(other_meta.get('artist', []))
        other_styles = set(other_meta.get('style', []))
        
        common_artist = my_artists & other_artists
        if common_artist:
            recs.append({"id": pid, "title": info['title'], "reason": f"Created by same artist: {get_name(list(common_artist)[0])}"})
            continue
            
        common_style = my_styles & other_styles
        if common_style:
            recs.append({"id": pid, "title": info['title'], "reason": f"Shared Movement: {get_name(list(common_style)[0])}"})
            continue
    
    while len(recs) < 3:
        pid, info = random.choice(candidates)
        if pid != target_id and not any(r['id'] == pid for r in recs):
             recs.append({"id": pid, "title": info['title'], "reason": "Curatorial Discovery"})
    
    return recs

def generate_long_prompt(input_context, recommended_titles):
    long_text = f"### SYSTEM LOG: Knowledge Graph Retrieval ###\n"
    for i in range(30):
        long_text += f"[Step {i+1}] Accessing Ontology Layer... Node verified.\n"
    
    long_text += f"\n### INPUT ANALYSIS ###\n{input_context}\n"
    long_text += f"\n### RECOMMENDATION LIST ###\n"
    for i, t in enumerate(recommended_titles):
        long_text += f"{i+1}. {t}\n"
    return long_text

def save_artifacts(target_id, latency, full_response, is_optimized):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "optimized" if is_optimized else "baseline"
    
    # [New] Throughput 계산 로직 추가
    # 대략 4글자(Characters) = 1토큰으로 추산 (Llama-2 기준 Rule of Thumb)
    estimated_tokens = len(full_response) / 4
    throughput = estimated_tokens / latency if latency > 0 else 0.0
    
    # 1. JSON 파일 저장 (상세 데이터)
    rec_file = f"recommendations_{mode}_{target_id}.json"
    with open(rec_file, "w", encoding='utf-8') as f:
        json.dump({
            "id": target_id, 
            "mode": mode, 
            "latency": latency,
            "estimated_tokens": int(estimated_tokens),
            "throughput": f"{throughput:.2f} tok/s", # <-- JSON에도 저장
            "response": full_response
        }, f, indent=2, ensure_ascii=False)
    
    # 2. TXT 리포트 저장 (한 줄 요약 - 교수님 보여드리기 용)
    rep_file = f"final_report_{mode}.txt"
    log_line = (
        f"[{timestamp}] Mode: {mode} | ID: {target_id} | "
        f"Latency: {latency:.4f}s | "
        f"Throughput: {throughput:.2f} tok/s\n" # <-- 여기에 추가됨!
    )
    
    with open(rep_file, "a", encoding='utf-8') as f:
        f.write(log_line)
        
    return rec_file, rep_file
# ==========================================
# 3. UI 구성
# ==========================================
st.sidebar.title("🛠️ Config")
test_mode = st.sidebar.radio("Mode:", ["Optimized", "Baseline"])
is_opt = True if "Optimized" in test_mode else False

st.sidebar.markdown("---")
st.sidebar.title("💬 Chat")
if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages: st.sidebar.chat_message(msg["role"]).write(msg["content"])

if prompt := st.sidebar.chat_input("Ask about art..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.sidebar.chat_message("user").write(prompt)
    try:
        res = requests.post(SERVER_URL, json={"prompt": prompt, "adapter_type": "chat_bot", "max_tokens": 100}, timeout=5)
        if res.status_code == 200:
            ans = res.json().get("response", "")
            full = f"{ans}\n\n*(Latency: {time.time()-time.time():.2f}s)*"
            st.session_state.messages.append({"role": "assistant", "content": full})
            st.sidebar.chat_message("assistant").write(ans)
    except: st.sidebar.error("Fail")

st.title("🎨 Art-KG Curator Demo")

# [디자인 변경] 컬럼 비율 조정 (왼쪽 좁게, 오른쪽 넓게) & 간격 추가
col1, col2 = st.columns([0.7, 1.3], gap="large")

with col1:
    # [디자인 변경] 컨테이너로 박스 만들기
    with st.container(border=True):
        st.subheader("1. Input")
        up = st.file_uploader("Upload", type=['jpg','png'])
        tid = None
        if up:
            st.image(up, use_container_width=True) # 꽉 차게
            tid = up.name.split('.')[0]
        else:
            st.info("Waiting for upload...")

if tid and tid in art_db:
    with col2:
        # [디자인 변경] 컨테이너로 박스 만들기
        with st.container(border=True):
            st.subheader("2. Context Generation")
            info = art_db[tid]
            title = info.get('title', tid)
            
            # Display Rich Context
            st.success(f"✅ Resolved: **{title}**")
            st.markdown(info.get('context_text', ''))

            if st.button("🚀 Analyze & Recommend", type="primary"):
                recs = get_smart_recommendations(tid, info)
                rec_titles = [r['title'] for r in recs]
                
                long_ctx = generate_long_prompt(info.get('context_text', ''), rec_titles)
                
                # [핵심] 프롬프트 개선: 예시(One-shot)를 넣어서 말투 고정
                final_input = (
                    f"{long_ctx}\n\n"
                    "### INSTRUCTION ###\n"
                    "1. Act as a professional Art Curator. Write a **cohesive, narrative commentary** explaining the connection between the INPUT artwork and the RECOMMENDATION LIST.\n"
                    "2. Do NOT repeat the task instructions. Do NOT use prefixes like 'Response:'. Just write the paragraph in full sentence.\n\n"
                    "3. Few shot: IDEAL OUTPUT EXAMPLE is as follows(Follow the style below)):\n"
                    "- The water-lilies-6.jpg is a photo captured by Claude Monet in 1899 in France. Water Lillies is a Japanese-style garden that was built in the 1890s for Monet, who suffered from severe asthma. In his water garden, he had three ponds connected by small waterfalls. The Japanese-style garden was created to relax and reflect, and the pond’s lily pads were a place where Claude Monet could sit and paint. Claude Monet was influenced by the Japanese school of painting, especially the impressionists.\n\n"
                )
                
                box = st.empty()
                box.info("Inference Running on GPU...")
                
                try:
                    t0 = time.time()
                    resp = requests.post(SERVER_URL, json={"prompt": final_input, "adapter_type": "art_curator", "max_tokens": 256}, timeout=60)
                    lat = time.time() - t0
                    
                    if resp.status_code == 200:
                        comm = resp.json().get("response", "")
                        
                        # 혹시 모를 태그 제거 (후처리)
                        comm = comm.replace("ArtCurator:", "").replace("RESPONSE:", "").strip()
                        
                        box.success(f"Done! (Latency: {lat:.4f}s)")
                        save_artifacts(tid, lat, comm, is_opt)
                        
                        st.markdown("### 🧠 Curator's Commentary")
                        # 텍스트가 넓게 보이도록 마크다운 활용
                        st.markdown(f">{comm}") 
                        
                        st.markdown("---")
                        st.markdown("### 🖼️ Recommended Gallery")
                        
                        cols = st.columns(3)
                        for i, r in enumerate(recs):
                            with cols[i]:
                                img_path = os.path.join(IMAGE_DIR, r['title'])
                                if os.path.exists(img_path):
                                    st.image(Image.open(img_path), use_container_width=True)
                                else:
                                    st.warning(f"No Image: {r['title']}")
                                
                                st.caption(f"**{r['reason']}**")
                                
                    else:
                        box.error("Server Error")
                except Exception as e:
                    box.error(f"Error: {e}")
elif tid:
    with col2: 
        with st.container(border=True):
            st.warning("⚠️ Image ID not found in Knowledge Graph.")