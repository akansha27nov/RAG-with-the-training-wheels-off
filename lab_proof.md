# Lab Proof: Native RAG with OpenAI & NumPy

## 1. Traceability Trace
* **User Query:** "What are the four ethical principles for Trustworthy AI?"
* **Top Similarity Score:** 0.7318
* **Retrieved Chunks Count (k):** 8
* **Top Matching Chunk:**
  > "amounts of digital data, major technologi cal advances in computational power and storage capacity, as well as  significant scientific and engineering innovation in AI methods and tools. AI systems will continue to impact society  and citizens in ways that we cannot yet imagine.  In this context, it is important to build AI systems that are worthy of trust, since human beings will only be able to  confidently and fully reap its benefits when the technology, including the processes and people behind the  technology, are trustworthy. When drafting these G uidelines, Trustworthy AI has, therefore, been our foundational  ambition.  Trustworthy AI has thre e components: (1) it should be l awful, ensuring compliance with  all applicable laws and  regulations, (2) it should be e thical, ensuring adherence to ethical principles and values and (3) it should be r obust,  both from a technical and social perspective since to ensure that, even with good intentions, AI systems do not  cause any unintentional harm. Each component is necessary but not sufficient to achieve Trustworthy AI. Ideally, all"
* **Final Grounded Answer:**
  > "The four ethical principles for Trustworthy AI are: 
(i) Respect for human autonomy 
(ii) Prevention of harm 
(iii) Fairness 
(iv) Explicability"

---

## 2. Failure Case & Limitations Analysis
**Question:** *Where could the system produce a plausible but unsupported answer or fail to retrieve?*

**Answer:** 
1. **Chunk Boundary Fragmentation:** As demonstrated during development, inadequate chunk overlap (e.g., 50 characters) can sever enumerated lists across chunk boundaries. If Principle (i) is cut into Chunk A and Principles (ii)-(iv) are in Chunk B, dense search may only retrieve Chunk B. Strict prompt guardrails will then cause the LLM to output "I don't know" because the context only contains 3 of the 4 principles.
2. **Dense Search Saturation:** In long documents, preamble and executive summary sections heavily reuse primary keyphrases (e.g., "Trustworthy AI"), filling up top vector ranks and pushing specific body text sections further down.

**Mitigation:** Increasing `chunk_overlap` (e.g., 200 characters) prevents list fragmentation, while hybrid search (BM25 + Dense) ensures specific structural sections are surfaced reliably.
