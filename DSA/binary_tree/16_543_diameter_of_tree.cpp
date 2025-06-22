#include <algorithm>
#include <climits>
#include <vector>
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
    int solve(TreeNode * root , int & diameter){
        if(root == nullptr) return 0;
        int leftPath = solve(root->left,diameter);
        int rightPath = solve(root->right ,diameter);
        diameter = max(diameter , leftPath +rightPath  );
        //  for diameter we take max with left and right path because we here count the edges and not the nodes
        // so if we have left path of 2 and right path of 3 then diameter will be 2+3 =5
        // if we had to count the nodes then we would do diameter = max(diameter , leftPath +rightPath +1);
        return 1+ max(leftPath , rightPath);
    }
    int diameterOfBinaryTree(TreeNode* root) {
        int diameter =0;
        solve(root,diameter);
        return diameter;
    }
};