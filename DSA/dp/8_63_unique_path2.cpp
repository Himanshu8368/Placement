#include <vector>
using namespace std;
class Solution {
public:
    int solve(int i,int j , vector<vector<int>>& obstacleGrid,vector<vector<int>>& dp){
        if(i<0 || j<0) return 0;
        if( obstacleGrid[i][j]==1){
            return 0;
        }
        if(i==0 && j==0) return 1;
        if(dp[i][j]!=-1) return dp[i][j];
        int up = solve(i-1,j,obstacleGrid,dp);
        int left = solve(i,j-1,obstacleGrid,dp);
        return dp[i][j] = up +left ;
    }
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size();
        int n = obstacleGrid[0].size();
        vector<vector<int>> dp(m,vector<int>(n,-1));

        return solve(m-1,n-1,obstacleGrid,dp);
    }
};


// Tabulation approach
// Time complexity: O(m*n)
class Solution1 {
public:
    // int solve(int i,int j , vector<vector<int>>& obstacleGrid,vector<vector<int>>& dp){
    //     if(i<0 || j<0) return 0;
    //     if( obstacleGrid[i][j]==1){
    //         return 0;
    //     }
    //     if(i==0 && j==0) return 1;
    //     if(dp[i][j]!=-1) return dp[i][j];
    //     int up = solve(i-1,j,obstacleGrid,dp);
    //     int left = solve(i,j-1,obstacleGrid,dp);
    //     return dp[i][j] = up +left ;
    // }
    int solve(int m,int n , vector<vector<int>>& obstacleGrid,vector<vector<int>>& dp){
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(obstacleGrid[i][j]==1){
                    dp[i][j]=0;
                    continue;
                }
                if(i==0 && j==0){
                    dp[i][j]=1;
                    continue;
                }
                int up =0;
                int left =0;
                if(i>0) up= dp[i-1][j];  
                if(j>0)  left = dp[i][j-1];
                dp[i][j]= up+left;             
            }
        }
        return dp[m-1][n-1];
    }
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size();
        int n = obstacleGrid[0].size();
        vector<vector<int>> dp(m,vector<int>(n,-1));

        return solve(m,n,obstacleGrid,dp);
    }
};