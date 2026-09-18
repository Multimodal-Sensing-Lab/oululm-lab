"""Edit evidence, retrieve passages, and inspect the exact augmented prompt.

Retrieval is real TF-IDF cosine similarity. This script does NOT invent an LLM answer.
Use answer_with_model.py with a staged instruction model for actual generation.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sklearn.feature_extraction.text import TfidfVectorizer
from llm_workshop.teaching import common_parser, show

DOCUMENTS = '''Aurora Lab: the Neral instrument is stored in room K142.
Aurora Lab: the Talin instrument is stored in room K108.
Aurora Lab: the garden team meets on Friday.'''
QUESTION = 'Where is the Neral instrument stored?'


def retrieve(question, documents=DOCUMENTS, top_k=2):
    if not question.strip():
        raise ValueError('Enter a question.')
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError('top_k must be a positive integer.')
    # Unlike topic 006, retrieval changes the supplied evidence, not model weights.
    passages = [line.strip() for line in documents.splitlines() if line.strip()]
    if not passages:
        raise ValueError('Add at least one document passage (one per line).')
    vectorizer = TfidfVectorizer()  # Build a word-overlap representation for retrieval.
    matrix = vectorizer.fit_transform(passages)  # Represent passages; these are not LLM embeddings.
    query = vectorizer.transform([question])  # Represent the question using the same term vocabulary.
    scores = (matrix @ query.T).toarray().ravel()  # Cosine similarity: TF-IDF rows have unit norm.
    ranked = sorted(range(len(passages)), key=lambda i: (-scores[i], i))
    selected = [i for i in ranked[:top_k] if scores[i] > 0]  # Keep the highest-ranked nonzero matches.
    evidence = '\n'.join(f'[doc-{i+1}] {passages[i]}' for i in selected)
    prompt = ('Answer using only the evidence. Cite document IDs. If it does not support an answer, say UNKNOWN.\n\n'
              f'EVIDENCE:\n{evidence or "(No matching passage)"}\n\nQUESTION: {question}\nANSWER:')
    return {'passages': passages, 'scores': scores, 'selected': selected, 'prompt': prompt,
            'terms': vectorizer.get_feature_names_out(), 'document_vectors': matrix.toarray(),
            'query_vector': query.toarray()[0]}


def main():
    parser = common_parser(__doc__)
    parser.set_defaults(text=QUESTION)
    parser.add_argument('--documents', type=Path, help='UTF-8 text file, one passage per line')
    parser.add_argument('--top-k', type=int, default=2)
    args = parser.parse_args()
    result = retrieve(args.text, args.documents.read_text() if args.documents else DOCUMENTS, args.top_k)
    show('Document passages', result['passages'], args.step)
    show('TF-IDF vocabulary', result['terms'], args.step)
    show('Cosine scores', result['scores'], args.step)
    show('Selected indices (zero based)', result['selected'], args.step)
    show('Exact input for a later LLM call', result['prompt'], args.step)
    print('No weights changed. No answer was generated. Change K142 in the evidence and run again.')


if __name__ == '__main__':
    main()
