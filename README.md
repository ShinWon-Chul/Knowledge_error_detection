# Incorrect Triple Detection Using Knowledge Graph Embedding and Adaptive Clustering
해당 레파지토리는 [지식 그래프 임베딩 및 적응형 클러스터링을 활용한 오류 트리플 검출](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10475012&language=ko_KR)를 구현한 것입니다.  
해당 코드는 지식 그래프를 임베딩 하여 k-means 클러스터링 기반의 오류 검출은 수행합니다.
해당 코드에 대한 직관적인 파악은 [`Knowledge_error_detection/지식그래프 임베딩 및 적응형 클러스터링을 활용한 오류 트리플 검출.pdf`](https://github.com/ShinWon-Chul/Knowledge_error_detection/blob/main/%EC%A7%80%EC%8B%9D%EA%B7%B8%EB%9E%98%ED%94%84%20%EC%9E%84%EB%B2%A0%EB%94%A9%20%EB%B0%8F%20%EC%A0%81%EC%9D%91%ED%98%95%20%ED%81%B4%EB%9F%AC%EC%8A%A4%ED%84%B0%EB%A7%81%EC%9D%84%20%ED%99%9C%EC%9A%A9%ED%95%9C%20%EC%98%A4%EB%A5%98%20%ED%8A%B8%EB%A6%AC%ED%94%8C%20%EA%B2%80%EC%B6%9C.pdf)를 참조 하십시오.

## 데이터 형식
NTP의 데이터는 트리플(subject, relation, object)형식을 따릅니다.
DBpedia, FreeBase, WiseKB 세가지 데이터는 실세계의 사람, 도시, 나라, 학교, 스포츠팀, 영화와 같은 개체들과 해당 개체들간의 관계를 나타냅니다. 데이터는 용량관계상 인코딩 되어있는 형식입니다.  

```shell
data/dbpedia/train_positive_20000.txt
e_2398311 r_275 e_651539
e_4856915 r_275 e_651539
e_375534 r_275 e_651539
e_401503 r_275 e_651539
e_1890861 r_275 e_651539
```

- `glove_*` 폴더는 `*`에 해당하는 데이터를 중심으로부터 ***sub-knowledge graph에 대한 문장***(자세한 내용은 [`Knowledge_error_detection/지식그래프 임베딩 및 적응형 클러스터링을 활용한 오류 트리플 검출.pdf`](https://github.com/ShinWon-Chul/Knowledge_error_detection/blob/main/%EC%A7%80%EC%8B%9D%EA%B7%B8%EB%9E%98%ED%94%84%20%EC%9E%84%EB%B2%A0%EB%94%A9%20%EB%B0%8F%20%EC%A0%81%EC%9D%91%ED%98%95%20%ED%81%B4%EB%9F%AC%EC%8A%A4%ED%84%B0%EB%A7%81%EC%9D%84%20%ED%99%9C%EC%9A%A9%ED%95%9C%20%EC%98%A4%EB%A5%98%20%ED%8A%B8%EB%A6%AC%ED%94%8C%20%EA%B2%80%EC%B6%9C.pdf) 참조)을 생성하여[GloVe](https://nlp.stanford.edu/projects/glove/) 알고리즘을 활용하여 임베딩한 벡터 결과를 나타냅니다.
	- `glove_dbpedia/person_embedding(numpy array)` : pickle로 저장된 (vocabsize,embedding_size)형식의 matrix
	- `glove_dbpedia/person_words(list)` : person_embedding matrix의 행 label word

- `skipgram_*` 폴더는 `*`에 해당하는 데이터를 중심으로부터 ***sub-knowledge graph에 대한 문장***을 생성 [Skip-gram](https://arxiv.org/pdf/1301.3781.pdf) 알고리즘을 활용하여 임베딩한 벡터 결과를 나타냅니다.

## Running
Skip-gram 기반 오류 검출 실행 파일은 [`Skip-gram_embedding_based_error_triple_detection.ipynb`](https://github.com/ShinWon-Chul/Knowledge_error_detection/blob/main/Skip-gram_embedding_based_error_triple_detection.ipynb) 입니다.
GloVe 임베딩 기반 오류 검출 실행 파일은 [`GloVe_embedding_based_error_triple_detection.ipynb`](https://github.com/ShinWon-Chul/Knowledge_error_detection/blob/main/GloVe_embedding_based_error_triple_detection.ipynb) 입니다.


## parameters
- 논문과 부합하는 결과를 얻기 위한 k-means 클러스터링의 optimal k
	- GloVe
		- DBpedia = 15
		- WiseKB = 27
		- FreeBase = 46
	- Skip-gram
		- DBpedia = 21
		- WiseKB = 27
		- FreeBase = 21