#include <vector>
using namespace std;

class Solution {
public:

    void dfs(int node ,vector<vector<int>> & adjList, vector<int> &visited){
        visited[node] = 1;
        for(auto it :adjList[node] ){
            if(!visited[it]){
                dfs(it,adjList,visited);
            }
        }
    }
    int findCircleNum(vector<vector<int>>& isConnected) {
        int v = isConnected.size();
        vector<int> visited(v,0);
        vector<vector<int>>adjList(v);
        for(int i=0 ;i<v;i++){
            for(int j=0;j<v;j++){
                if(i!=j && isConnected[i][j]==1){
                    adjList[i].push_back(j);
                    adjList[j].push_back(i);
                }
            }
        }
        int count =0;
        for(int i=0 ;i<v;i++){
            if(!visited[i]){
                dfs(i,adjList,visited);
                count++;
            }
        }
        return count;

    }
};