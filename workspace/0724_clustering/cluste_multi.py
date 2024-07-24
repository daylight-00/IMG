import pandas as pd
from Bio.Align import PairwiseAligner
import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# 임의의 CSV 파일 읽기
file_path = 'hla_counts/mhc_ligand_clean_final_split.csv'
df = pd.read_csv(file_path)

# 'Epitope' 열의 정보 가져오기
sequences = df['Epitope'].dropna().unique()

# 서열 유사도 계산 함수
def calculate_similarity(args):
    seq1, seq2 = args
    aligner = PairwiseAligner()
    alignments = aligner.align(seq1, seq2)
    score = alignments[0].score
    return (seq1, seq2, score / max(len(seq1), len(seq2)))

# 유사도 행렬 생성 (멀티 프로세싱 적용)
def create_similarity_matrix(sequences, num_cores, chunk_size):
    with Pool(num_cores) as pool:
        similarity_matrix = {}
        args = [(seq1, seq2) for i, seq1 in enumerate(sequences) for j, seq2 in enumerate(sequences) if i < j]
        
        for i in range(0, len(args), chunk_size):
            chunk_args = args[i:i + chunk_size]
            results = pool.map(calculate_similarity, chunk_args)
            for result in results:
                seq1, seq2, similarity = result
                similarity_matrix[(seq1, seq2)] = similarity

    return similarity_matrix

if __name__ == '__main__':
    num_cores = cpu_count()  # 사용할 코어 수 설정
    chunk_size = 500  # 청크 크기 설정 (적절히 조정 필요)
    similarity_matrix = create_similarity_matrix(sequences, num_cores, chunk_size)

    # NetworkX 그래프 생성
    G = nx.Graph()
    for (seq1, seq2), similarity in similarity_matrix.items():
        if similarity > 0.7:  # 유사도 임계값 설정
            G.add_edge(seq1, seq2, weight=similarity)

    # 클러스터링
    clusters = list(nx.connected_components(G))

    # 결과 출력
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}: {cluster}")

    # 클러스터 시각화
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=10, font_weight="bold")
    plt.title("Epitope Clustering")
    plt.show()
