#include<vector>
#include<algorithm>
#include<climits>
using namespace std;
struct TreeNode {
     int val;
     TreeNode *left;
     TreeNode *right;
     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 };
class Solution {
public:
    int solve(TreeNode * root , int & max_sum){
        if(root == nullptr){
            return 0;
        }
        int left_sum = max(0,solve(root->left,max_sum));
        int right_sum = max(0,solve(root->right,max_sum));
        max_sum = max(max_sum , left_sum + right_sum+root->val);

        // If we want to return the maximum path sum that can be extended to the parent node,
        // we return the maximum of left_sum and right_sum plus the current node's value.
        // This is because we can only extend one side of the path to the parent.
        // If we return both sides, it would not be a valid path in the context of the problem.
        return (max(left_sum ,right_sum)+root->val);
    }
    int maxPathSum(TreeNode* root) {
        int max_sum =INT_MIN;
        solve(root,max_sum);
        return max_sum;
    }
};