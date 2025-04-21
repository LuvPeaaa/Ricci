# Import các thư viện cần thiết
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, cosine
from sklearn.preprocessing import StandardScaler
from dtaidistance import dtw

# 1. Tạo dữ liệu chuỗi thời gian mẫu
# Giả sử chúng ta có dữ liệu giá cổ phiếu trong 100 ngày
np.random.seed(42)  # Đặt seed cho tính tái tạo
n_days = 100

# Tạo ra một số chuỗi thời gian đơn giản (giá đóng cửa, khối lượng, biến động)
close_prices = np.cumsum(np.random.normal(0, 1, n_days)) + 100
volumes = np.random.normal(1000, 200, n_days)
volatility = np.abs(np.random.normal(0, 1, n_days))

# Tạo DataFrame từ dữ liệu
data = pd.read_csv('/data/stock_data.csv')

print(f"Dữ liệu ban đầu: {data.shape} dòng")

# 2. Trích xuất đặc trưng sử dụng sliding window theo cách của bạn
def extract_ft(time_series_data, window_size=10):
    """
    Trích xuất đặc trưng từ chuỗi thời gian sử dụng sliding window
    
    Tham số:
    - time_series_data: Dữ liệu chuỗi thời gian một chiều
    - window_size: Kích thước cửa sổ trượt
    
    Trả về:
    - Mảng các vector đặc trưng tương ứng với mỗi điểm thời gian
    """
    features = []
    
    # Trích xuất đặc trưng:
    # - Trung bình cửa sổ trượt
    # - Độ lệch chuẩn cửa sổ trượt
    # - Min/Max trong cửa sổ
    for i in range(len(time_series_data) - window_size + 1):
        window = time_series_data[i:i + window_size]
        mean = np.mean(window)
        std = np.std(window)
        min_val = np.min(window)
        max_val = np.max(window)
        feature_vector = [mean, std, min_val, max_val]  # Sửa lỗi ở đây
        features.append(feature_vector)
    
    return np.array(features)

# 3. Mở rộng để xử lý nhiều chuỗi thời gian
def extract_all_features(data, window_size=10):
    """
    Trích xuất đặc trưng từ tất cả các chuỗi thời gian trong DataFrame
    
    Tham số:
    - data: DataFrame chứa nhiều chuỗi thời gian
    - window_size: Kích thước cửa sổ trượt
    
    Trả về:
    - List các vector đặc trưng cho mỗi điểm thời gian hợp lệ
    """
    # Lấy các đặc trưng cho từng cột dữ liệu
    close_features = extract_ft(data['close'].values, window_size)
    volume_features = extract_ft(data['volume'].values, window_size)
    volatility_features = extract_ft(data['volatility'].values, window_size)
    
    # Số lượng điểm thời gian sau khi áp dụng sliding window
    n_points = len(close_features)
    
    # Tạo list lưu trữ các vector đặc trưng tổng hợp
    all_features = []
    
    # Kết hợp các đặc trưng từ tất cả các chuỗi thời gian
    for i in range(n_points):
        # Gộp các đặc trưng từ các chuỗi khác nhau
        combined_feature = np.concatenate([
            close_features[i],
            volume_features[i],
            volatility_features[i]
        ])
        all_features.append(combined_feature)
    
    return all_features

# 4. Trích xuất đặc trưng từ tất cả các chuỗi thời gian
window_size = 10  # Kích thước cửa sổ trượt
feature_vectors = extract_all_features(data, window_size)

# Số điểm thời gian có đặc trưng (giảm do sliding window)
n_feature_points = len(feature_vectors)
print(f"Số điểm thời gian có vector đặc trưng: {n_feature_points}")
print(f"Kích thước mỗi vector đặc trưng: {len(feature_vectors[0])}")

# 5. Chuẩn hóa các vector đặc trưng
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_vectors)

# 6. Tính khoảng cách giữa các vector đặc trưng
def compute_distance(feature_vector1, feature_vector2, method='euclidean'):
    """
    Tính khoảng cách giữa hai vector đặc trưng
    
    Tham số:
    - feature_vector1, feature_vector2: Các vector đặc trưng cần so sánh
    - method: Phương pháp tính khoảng cách ('euclidean', 'cosine', 'dtw')
    
    Trả về:
    - Giá trị khoảng cách
    """
    if method == 'euclidean':
        return euclidean(feature_vector1, feature_vector2)
    elif method == 'cosine':
        return cosine(feature_vector1, feature_vector2)
    elif method == 'dtw':
        return dtw.distance(feature_vector1.reshape(-1, 1), feature_vector2.reshape(-1, 1))
    else:
        raise ValueError("Phương pháp không được hỗ trợ. Hãy chọn 'euclidean', 'cosine', hoặc 'dtw'")

# 7. Xây dựng đồ thị dựa trên ngưỡng tương đồng
def build_similarity_graph(features, epsilon=2.0, distance_method='euclidean'):
    """
    Xây dựng đồ thị dựa trên độ tương đồng của các vector đặc trưng
    
    Tham số:
    - features: Danh sách các vector đặc trưng đã chuẩn hóa
    - epsilon: Ngưỡng khoảng cách để tạo cạnh
    - distance_method: Phương pháp tính khoảng cách
    
    Trả về:
    - Đồ thị NetworkX
    """
    G = nx.Graph()
    
    # Thêm tất cả các điểm thời gian làm nút
    for t in range(len(features)):
        G.add_node(t)
    
    # Thêm cạnh nếu khoảng cách nhỏ hơn epsilon
    for t in range(len(features)):
        for j in range(t+1, len(features)):
            f_t = features[t]
            f_j = features[j]
            
            distance = compute_distance(f_t, f_j, method=distance_method)
            
            # Nếu khoảng cách nhỏ hơn epsilon, thêm cạnh
            if distance < epsilon:
                # Lưu khoảng cách làm trọng số của cạnh
                G.add_edge(t, j, weight=distance)
    
    return G

# 8. Xây dựng đồ thị với khoảng cách Euclidean
# Có thể điều chỉnh epsilon để có mật độ cạnh phù hợp
graph = build_similarity_graph(scaled_features, epsilon=2.5, distance_method='euclidean')
print(f"Đồ thị có {graph.number_of_nodes()} nút và {graph.number_of_edges()} cạnh")

# 9. Tính Ricci Curvature trên đồ thị
def compute_ricci_curvature(G):
    """
    Tính đại lượng gần đúng với Ollivier-Ricci curvature cho mỗi cạnh
    
    Tham số:
    - G: NetworkX graph
    
    Trả về:
    - Dictionary với keys là cạnh và values là giá trị curvature
    """
    curvatures = {}
    
    for edge in G.edges():
        i, j = edge
        
        # Lấy neighbors (láng giềng) của mỗi nút, không bao gồm nút kia
        i_neighbors = set(G.neighbors(i)) - {j}
        j_neighbors = set(G.neighbors(j)) - {i}
        
        # Nếu không có đủ neighbors, gán giá trị curvature = 0
        if not i_neighbors or not j_neighbors:
            curvatures[edge] = 0
            continue
        
        # Tính khoảng cách trung bình từ neighbors của i đến neighbors của j
        # Đây là một phép tính đơn giản hóa của khoảng cách Wasserstein
        sum_distance = 0
        count = 0
        
        for ni in i_neighbors:
            for nj in j_neighbors:
                # Sử dụng khoảng cách trên đồ thị (số cạnh tối thiểu)
                if G.has_edge(ni, nj):
                    # Nếu có cạnh trực tiếp, sử dụng trọng số
                    distance = G[ni][nj]['weight']
                else:
                    # Nếu không có cạnh trực tiếp, sử dụng giá trị lớn
                    try:
                        # Tìm đường đi ngắn nhất
                        distance = nx.shortest_path_length(G, source=ni, target=nj, weight='weight')
                    except nx.NetworkXNoPath:
                        # Nếu không có đường đi, gán giá trị lớn
                        distance = float('inf')
                
                if distance != float('inf'):
                    sum_distance += distance
                    count += 1
        
        if count > 0:
            avg_distance = sum_distance / count
        else:
            avg_distance = float('inf')
        
        # Tính khoảng cách trực tiếp giữa i và j
        direct_distance = G[i][j]['weight']
        
        # Tính curvature: 1 - (avg_distance / direct_distance)
        # Giá trị dương: không gian cong dương (như mặt cầu)
        # Giá trị âm: không gian cong âm (như hình hyperbolic)
        # Giá trị 0: không gian phẳng (Euclidean)
        if direct_distance > 0 and avg_distance != float('inf'):
            curvature = 1 - (avg_distance / direct_distance)
        else:
            curvature = 0
        
        curvatures[edge] = curvature
    
    return curvatures

# 10. Tính Ricci curvature
ricci_curvatures = compute_ricci_curvature(graph)

# In một số kết quả
print("\nRicci curvature cho một số cạnh:")
for i, (edge, curvature) in enumerate(ricci_curvatures.items()):
    print(f"Cạnh {edge}: {curvature:.4f}")
    if i >= 4:  # In 5 kết quả đầu tiên
        break

# 11. Phân tích đồ thị dựa trên độ cong Ricci
def analyze_curvature(curvatures):
    """Phân tích các giá trị curvature"""
    curvature_values = list(curvatures.values())
    
    print(f"\nThống kê độ cong Ricci:")
    print(f"Số cạnh: {len(curvature_values)}")
    print(f"Giá trị trung bình: {np.mean(curvature_values):.4f}")
    print(f"Giá trị nhỏ nhất: {np.min(curvature_values):.4f}")
    print(f"Giá trị lớn nhất: {np.max(curvature_values):.4f}")
    
    # Đếm số cạnh có độ cong dương, âm và zero
    pos_curvature = sum(1 for c in curvature_values if c > 0)
    neg_curvature = sum(1 for c in curvature_values if c < 0)
    zero_curvature = sum(1 for c in curvature_values if c == 0)
    
    print(f"Số cạnh có độ cong dương: {pos_curvature} ({pos_curvature/len(curvature_values)*100:.1f}%)")
    print(f"Số cạnh có độ cong âm: {neg_curvature} ({neg_curvature/len(curvature_values)*100:.1f}%)")
    print(f"Số cạnh có độ cong bằng 0: {zero_curvature} ({zero_curvature/len(curvature_values)*100:.1f}%)")

# 12. Phân tích độ cong
analyze_curvature(ricci_curvatures)

# 13. Hiển thị đồ thị với các màu thể hiện độ cong Ricci
def visualize_graph_with_curvature(G, curvatures):
    """
    Hiển thị đồ thị với các cạnh được tô màu theo độ cong Ricci
    
    Tham số:
    - G: NetworkX graph
    - curvatures: Dictionary chứa giá trị curvature cho mỗi cạnh
    """
    plt.figure(figsize=(12, 8))
    
    # Tạo layout cho đồ thị
    pos = nx.spring_layout(G, seed=42)
    
    # Vẽ các nút
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightblue')
    
    # Vẽ các cạnh với màu sắc thể hiện độ cong
    # Đỏ: cong âm, Xanh lá: cong dương, Xám: cong ≈ 0
    for edge, curvature in curvatures.items():
        if curvature > 0.01:  # Cong dương
            color = 'green'
            width = 1 + 2 * curvature  # Càng cong dương càng dày
        elif curvature < -0.01:  # Cong âm
            color = 'red'
            width = 1 + 2 * abs(curvature)  # Càng cong âm càng dày
        else:  # Gần như phẳng
            color = 'gray'
            width = 0.5
        
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=width, edge_color=color, alpha=0.7)
    
    # Vẽ labels cho một số nút (ngày)
    # Chỉ hiển thị một số nhãn để tránh quá tải
    selected_nodes = list(range(0, len(G.nodes()), 10))  # Mỗi 10 điểm
    labels = {node: f'T{node}' for node in selected_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    plt.title('Đồ thị tương đồng với độ cong Ricci (Phương pháp Sliding Window)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 14. Hiển thị đồ thị
visualize_graph_with_curvature(graph, ricci_curvatures)

# 15. Phân tích cụm dựa trên đồ thị và độ cong Ricci
def analyze_clusters(G, curvatures):
    """
    Phân tích các cụm trong đồ thị dựa trên độ cong Ricci
    
    Tham số:
    - G: NetworkX graph
    - curvatures: Dictionary chứa giá trị curvature cho mỗi cạnh
    """
    # Tạo đồ thị mới chỉ với các cạnh có độ cong dương
    G_pos = nx.Graph()
    G_pos.add_nodes_from(G.nodes())
    
    # Thêm các cạnh có độ cong dương
    for edge, curvature in curvatures.items():
        if curvature > 0:
            G_pos.add_edge(edge[0], edge[1])
    
    # Tìm các thành phần liên thông (cụm)
    clusters = list(nx.connected_components(G_pos))
    
    print(f"\nPhân tích cụm dựa trên độ cong Ricci dương:")
    print(f"Số lượng cụm: {len(clusters)}")
    
    # In thông tin về các cụm lớn nhất
    sorted_clusters = sorted(clusters, key=len, reverse=True)
    for i, cluster in enumerate(sorted_clusters[:3]):  # 3 cụm lớn nhất
        print(f"Cụm {i+1}: {len(cluster)} nút")
        print(f"  Các điểm thời gian trong cụm: {sorted(list(cluster))[:10]}...")
    
    return clusters

# 16. Phân tích cụm
clusters = analyze_clusters(graph, ricci_curvatures)

# 17. Hiển thị kết quả phân đoạn chuỗi thời gian
def visualize_time_series_clusters(data, clusters, window_size):
    """
    Hiển thị kết quả phân đoạn chuỗi thời gian dựa trên các cụm
    
    Tham số:
    - data: DataFrame chứa dữ liệu chuỗi thời gian
    - clusters: Danh sách các cụm (mỗi cụm là tập hợp các chỉ số)
    - window_size: Kích thước cửa sổ trượt đã sử dụng
    """
    # Điều chỉnh chỉ số từ sau khi sử dụng sliding window về dữ liệu gốc
    plt.figure(figsize=(12, 6))
    
    # Vẽ chuỗi thời gian gốc
    plt.plot(data['close'], color='gray', alpha=0.5, label='Giá đóng cửa')
    
    # Vẽ các cụm lớn nhất với màu khác nhau
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    for i, cluster in enumerate(sorted(clusters, key=len, reverse=True)[:5]):  # 5 cụm lớn nhất
        if len(cluster) < 2:  # Bỏ qua cụm quá nhỏ
            continue
            
        # Điều chỉnh chỉ số từ cửa sổ trượt về dữ liệu gốc
        adjusted_indices = []
        for idx in cluster:
            # Mỗi chỉ số idx trong không gian cửa sổ trượt
            # tương ứng với điểm thời gian idx -> idx+window_size-1 trong dữ liệu gốc
            adjusted_indices.extend(range(idx, idx + window_size))
        
        # Loại bỏ các chỉ số trùng lặp và đảm bảo nằm trong phạm vi dữ liệu
        adjusted_indices = sorted(set(adjusted_indices))
        adjusted_indices = [idx for idx in adjusted_indices if idx < len(data)]
        
        if adjusted_indices:
            plt.scatter(adjusted_indices, data.loc[adjusted_indices, 'close'], color=colors[i % len(colors)], label=f'Cụm {i+1}', alpha=0.7)
    
    plt.title('Phân đoạn chuỗi thời gian dựa trên độ cong Ricci')
    plt.xlabel('Thời gian')
    plt.ylabel('Giá trị')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 18. Hiển thị kết quả phân đoạn
visualize_time_series_clusters(data, clusters, window_size)

# 19. Phân tích đặc điểm của từng cụm
def analyze_cluster_features(data, clusters, window_size):
    """
    Phân tích đặc điểm của từng cụm trong dữ liệu chuỗi thời gian
    
    Tham số:
    - data: DataFrame chứa dữ liệu gốc
    - clusters: List các tập hợp nút tạo thành cụm
    - window_size: Kích thước cửa sổ trượt đã sử dụng
    """
    print("\nPhân tích đặc điểm của các cụm (từ sliding window):")
    
    # Lấy các cụm lớn nhất
    sorted_clusters = sorted(clusters, key=len, reverse=True)
    
    for i, cluster in enumerate(sorted_clusters[:3]):  # 3 cụm lớn nhất
        if len(cluster) == 0:
            continue
        
        # Điều chỉnh chỉ số từ cửa sổ trượt về dữ liệu gốc
        adjusted_indices = []
        for idx in cluster:
            # Mỗi chỉ số idx trong không gian cửa sổ trượt
            # tương ứng với điểm thời gian idx -> idx+window_size-1 trong dữ liệu gốc
            adjusted_indices.extend(range(idx, idx + window_size))
        
        # Loại bỏ các chỉ số trùng lặp và đảm bảo nằm trong phạm vi dữ liệu
        adjusted_indices = sorted(set(adjusted_indices))
        adjusted_indices = [idx for idx in adjusted_indices if idx < len(data)]
        
        if not adjusted_indices:
            continue
            
        cluster_data = data.iloc[adjusted_indices]
        
        print(f"\nCụm {i+1} ({len(cluster)} cửa sổ, {len(adjusted_indices)} điểm thời gian):")
        print(f"  Giá trung bình: {cluster_data['close'].mean():.2f}")
        print(f"  Khối lượng trung bình: {cluster_data['volume'].mean():.2f}")
        print(f"  Biến động trung bình: {cluster_data['volatility'].mean():.2f}")
        
        # So sánh với toàn bộ dữ liệu
        price_diff = (cluster_data['close'].mean() / data['close'].mean() - 1) * 100
        volume_diff = (cluster_data['volume'].mean() / data['volume'].mean() - 1) * 100
        vol_diff = (cluster_data['volatility'].mean() / data['volatility'].mean() - 1) * 100
        
        print(f"  So với toàn bộ dữ liệu:")
        print(f"    Giá: {price_diff:+.2f}%")
        print(f"    Khối lượng: {volume_diff:+.2f}%")
        print(f"    Biến động: {vol_diff:+.2f}%")
        
        # Phân tích xu hướng trong cụm
        if len(adjusted_indices) > 1:
            cluster_trend = np.polyfit(range(len(cluster_data)), cluster_data['close'], 1)[0]
            if cluster_trend > 0:
                trend_desc = f"Xu hướng tăng (+{cluster_trend:.4f}/đơn vị thời gian)"
            else:
                trend_desc = f"Xu hướng giảm ({cluster_trend:.4f}/đơn vị thời gian)"
            print(f"  {trend_desc}")

# 20. Phân tích đặc điểm của các cụm
analyze_cluster_features(data, clusters, window_size)