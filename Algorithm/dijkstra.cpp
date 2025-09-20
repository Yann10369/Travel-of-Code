#include <iostream>
#include <vector>
#include <queue>
#include <climits>

using namespace std;

// 定义边的结构体，包含目标节点和边权
struct Edge {
    int to;
    int weight;
    Edge(int t, int w) : to(t), weight(w) {}
}; 

// Dijkstra算法实现
vector<int> dijkstra(const vector<vector<Edge>>& graph, int start) {
    int n = graph.size();
    vector<int> dist(n, INT_MAX);  // 存储起点到各节点的最短距离
    vector<bool> visited(n, false);  // 标记节点是否已访问
    
    // 优先队列，存储{距离, 节点}对，按距离从小到大排序
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;//函数模板，后面是类型
    
    // 起点到自身的距离为0
    dist[start] = 0;
    pq.push({0, start});
    
    while (!pq.empty()) {
        // 取出当前距离最小的节点
        int u = pq.top().second;
        pq.pop();
        
        // 如果节点已访问，跳过
        if (visited[u]) continue;
        visited[u] = true;
        
        // 遍历所有邻接边
        for (const Edge& edge : graph[u]) {
            int v = edge.to;
            int weight = edge.weight;
            
            // 松弛操作
            if (!visited[v] && dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}

int main() {
    // 示例：构建一个包含5个节点的图
    int n = 5;
    vector<vector<Edge>> graph(n);
    
    // 添加边 (节点0到节点1的距离为4)
    graph[0].emplace_back(1, 4);
    graph[0].emplace_back(2, 1);
    graph[1].emplace_back(2, 2);
    graph[1].emplace_back(3, 5);
    graph[2].emplace_back(1, 2);
    graph[2].emplace_back(3, 8);
    graph[2].emplace_back(4, 10);
    graph[3].emplace_back(4, 2);
    graph[4].emplace_back(3, 2);
    
    // 计算从节点0出发到各节点的最短路径
    vector<int> shortestPaths = dijkstra(graph, 0);
    
    // 输出结果
    cout << "从节点0到各节点的最短路径：" << endl;
    for (int i = 0; i < n; ++i) {
        cout << "到节点" << i << "的最短距离是：";
        if (shortestPaths[i] == INT_MAX) {
            cout << "无穷大" << endl;
        } else {
            cout << shortestPaths[i] << endl;
        }
    }
    
    return 0;
}