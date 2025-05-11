import streamlit as st
import torch
from transformers import T5Tokenizer, MT5ForConditionalGeneration, PegasusTokenizer, PegasusForConditionalGeneration
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

@st.cache_resource
def load_pegasus():
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_mt5_xlsum():
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def summarize_model(text, tokenizer, model, prefix=""):
    input_text = prefix + text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="longest", max_length=512)
    with torch.no_grad():
        summary_ids = model.generate(inputs["input_ids"], max_length=60, min_length=20, do_sample=False)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def compute_bleu(reference, prediction):
    smoothie = SmoothingFunction().method4
    return round(sentence_bleu([reference.split()], prediction.split(), smoothing_function=smoothie), 4)

rouge = evaluate.load("rouge")

def show_summarization():
    st.title("📄 LLM Summarization Comparison: mT5_XLSum vs Pegasus")

    input_text = st.text_area("✍️ Enter Policy or News Text:", height=250)
    reference_summary = st.text_area("📌 Reference Summary (for ROUGE/BLEU evaluation):", height=150)

    if st.button("🔍 Compare Summarizers"):
        if not input_text.strip():
            st.warning("Please provide input text.")
            return

        peg_tokenizer, peg_model = load_pegasus()
        mt5_tokenizer, mt5_model = load_mt5_xlsum()

        with st.spinner("Running mT5_XLSum..."):
            mt5_summary = summarize_model(input_text, mt5_tokenizer, mt5_model)

        with st.spinner("Running Pegasus..."):
            peg_summary = summarize_model(input_text, peg_tokenizer, peg_model)

        st.subheader("📝 mT5_XLSum Summary")
        st.info(mt5_summary)

        st.subheader("📝 Pegasus Summary")
        st.success(peg_summary)

        if reference_summary.strip():
            st.subheader("📊 Evaluation Metrics")

            r_mt5 = rouge.compute(predictions=[mt5_summary], references=[reference_summary])
            b_mt5 = compute_bleu(reference_summary, mt5_summary)
            st.markdown("**mT5_XLSum**")
            st.json(r_mt5)
            st.write(f"BLEU: `{b_mt5}`")

            r_peg = rouge.compute(predictions=[peg_summary], references=[reference_summary])
            b_peg = compute_bleu(reference_summary, peg_summary)
            st.markdown("**Pegasus**")
            st.json(r_peg)
            st.write(f"BLEU: `{b_peg}`")

        from sentence_transformers import SentenceTransformer, util

        @st.cache_resource
        def load_embedding_model():
            return SentenceTransformer("all-MiniLM-L6-v2")

        def highlight_rationale(input_text, summary):
            model = load_embedding_model()
            input_sents = [s.strip() for s in input_text.split('.') if len(s.strip()) > 20]
            summary_sents = [s.strip() for s in summary.split('.') if len(s.strip()) > 10]

            input_embeddings = model.encode(input_sents, convert_to_tensor=True)
            summary_embeddings = model.encode(summary_sents, convert_to_tensor=True)

            rationale_indices = set()
            for i, s_emb in enumerate(summary_embeddings):
                sims = util.cos_sim(s_emb, input_embeddings)[0]
                top_match = torch.argmax(sims).item()
                rationale_indices.add(top_match)

            return [input_sents[i] for i in rationale_indices]



