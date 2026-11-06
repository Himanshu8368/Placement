// Glossary Popup System
(function() {
    'use strict';
    
    // Glossary terms dictionary
    const glossary = {
        'activation function': 'A mathematical function applied to the output of a neuron to introduce non-linearity into the network.',
        'adversarial training': 'Training a model to be robust against adversarial examples - inputs designed to fool the model.',
        'attention mechanism': 'A technique that allows models to focus on relevant parts of the input when making predictions.',
        'autoencoder': 'A neural network architecture that learns to compress and reconstruct input data.',
        'backpropagation': 'An algorithm for training neural networks by propagating errors backward through the network.',
        'batch normalization': 'A technique that normalizes the inputs to each layer to stabilize training.',
        'bert': 'Bidirectional Encoder Representations from Transformers - a transformer-based language model.',
        'bias': 'Systematic error in model predictions, or a learnable parameter in neural networks.',
        'cnn': 'Convolutional Neural Network - a neural network architecture designed for processing grid-like data such as images.',
        'convolutional neural network': 'A neural network architecture designed for processing grid-like data such as images.',
        'cross-entropy loss': 'A loss function commonly used for classification tasks.',
        'curriculum learning': 'Training strategy where models learn from easy examples before harder ones.',
        'deep learning': 'A subset of machine learning using neural networks with multiple layers.',
        'dropout': 'A regularization technique that randomly sets some neurons to zero during training.',
        'decoder': 'The part of a sequence-to-sequence model that generates the output sequence.',
        'embedding': 'A dense vector representation of discrete objects like words or categories.',
        'encoder': 'The part of a sequence-to-sequence model that processes the input sequence.',
        'epoch': 'One complete pass through the entire training dataset.',
        'feature engineering': 'The process of creating informative features from raw data.',
        'fine-tuning': 'Adapting a pre-trained model to a specific task by training on task-specific data.',
        'forward pass': 'The process of passing input data through a neural network to get predictions.',
        'gan': 'Generative Adversarial Network - a framework where two networks compete: a generator and a discriminator.',
        'generative adversarial network': 'A framework where two networks compete: a generator and a discriminator.',
        'gradient': 'The derivative of the loss function with respect to model parameters.',
        'gradient descent': 'An optimization algorithm that minimizes the loss function.',
        'hidden layer': 'A layer in a neural network between the input and output layers.',
        'hyperparameter': 'A parameter set before training that controls the learning process.',
        'inference': 'The process of making predictions using a trained model.',
        'initialization': 'Setting initial values for model parameters before training.',
        'k-means': 'An unsupervised clustering algorithm that partitions data into k clusters.',
        'kernel': 'A function used in support vector machines to map data into higher dimensions.',
        'lstm': 'Long Short-Term Memory - a type of recurrent neural network that can learn long-term dependencies.',
        'long short-term memory': 'A type of recurrent neural network that can learn long-term dependencies.',
        'learning rate': 'A hyperparameter that controls how much the model parameters are updated during training.',
        'loss function': 'A function that measures how well the model\'s predictions match the true values.',
        'machine learning': 'A method of data analysis that automates analytical model building.',
        'momentum': 'An optimization technique that accelerates gradient descent.',
        'neural network': 'A computing system inspired by biological neural networks.',
        'normalization': 'Scaling data to a standard range or distribution.',
        'overfitting': 'When a model learns the training data too well and performs poorly on new data.',
        'optimizer': 'An algorithm that updates model parameters to minimize the loss function.',
        'pooling': 'A downsampling operation that reduces the spatial dimensions of feature maps.',
        'precision': 'A metric measuring the proportion of positive predictions that are correct.',
        'pre-training': 'Training a model on a large dataset before fine-tuning on a specific task.',
        'rnn': 'Recurrent Neural Network - a neural network architecture designed for sequential data.',
        'recurrent neural network': 'A neural network architecture designed for sequential data.',
        'regularization': 'Techniques to prevent overfitting by constraining model complexity.',
        'relu': 'Rectified Linear Unit - a popular activation function that outputs the input if positive, else zero.',
        'rectified linear unit': 'A popular activation function that outputs the input if positive, else zero.',
        'sgd': 'Stochastic Gradient Descent - a variant of gradient descent that uses random subsets of data.',
        'stochastic gradient descent': 'A variant of gradient descent that uses random subsets of data.',
        'softmax': 'An activation function that converts logits into probability distributions.',
        'supervised learning': 'Learning from labeled training data.',
        'transformer': 'A neural network architecture based on self-attention mechanisms.',
        'transfer learning': 'Using knowledge from one task to improve performance on another task.',
        'underfitting': 'When a model is too simple to capture the underlying patterns in the data.',
        'unsupervised learning': 'Learning patterns from unlabeled data.',
        'validation set': 'A subset of data used to tune hyperparameters and evaluate model performance.',
        'vanishing gradient': 'A problem where gradients become very small, preventing effective learning.',
        'weight': 'A learnable parameter in a neural network that multiplies inputs.',
        'word embedding': 'A dense vector representation of words learned from text data.',
        'xgboost': 'An optimized gradient boosting library for machine learning.',
        'zero-shot learning': 'Making predictions on classes not seen during training.'
    };
    
    let tooltip = null;
    let currentTerm = null;
    
    function createTooltip() {
        if (tooltip) return tooltip;
        
        tooltip = document.createElement('div');
        tooltip.className = 'glossary-tooltip';
        tooltip.style.cssText = `
            position: absolute;
            background: var(--md-default-bg-color);
            border: 1px solid var(--md-default-fg-color--lighter);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            z-index: 10000;
            max-width: 300px;
            font-size: 0.875rem;
            line-height: 1.5;
            display: none;
            pointer-events: none;
        `;
        document.body.appendChild(tooltip);
        return tooltip;
    }
    
    function showTooltip(term, definition, element) {
        const tip = createTooltip();
        tip.innerHTML = `
            <div style="font-weight: 600; margin-bottom: 0.5rem; color: var(--md-primary-fg-color);">
                ${term.charAt(0).toUpperCase() + term.slice(1)}
            </div>
            <div style="color: var(--md-default-fg-color);">
                ${definition}
            </div>
        `;
        tip.style.display = 'block';
        
        const rect = element.getBoundingClientRect();
        const tipRect = tip.getBoundingClientRect();
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;
        
        // Position tooltip above or below the element
        let top = rect.top + scrollTop - tipRect.height - 10;
        let left = rect.left + scrollLeft + (rect.width / 2) - (tipRect.width / 2);
        
        // Adjust if tooltip goes off screen
        if (top < scrollTop) {
            top = rect.bottom + scrollTop + 10;
        }
        if (left < 0) {
            left = 10;
        }
        if (left + tipRect.width > window.innerWidth) {
            left = window.innerWidth - tipRect.width - 10;
        }
        
        tip.style.top = top + 'px';
        tip.style.left = left + 'px';
    }
    
    function hideTooltip() {
        if (tooltip) {
            tooltip.style.display = 'none';
        }
        currentTerm = null;
    }
    
    function findGlossaryTerms() {
        const article = document.querySelector('.md-content__inner') || document.querySelector('article');
        if (!article) return;
        
        const textNodes = [];
        const walker = document.createTreeWalker(
            article,
            NodeFilter.SHOW_TEXT,
            {
                acceptNode: function(node) {
                    // Skip code blocks and certain elements
                    const parent = node.parentElement;
                    if (parent.tagName === 'CODE' || 
                        parent.tagName === 'PRE' || 
                        parent.closest('pre') ||
                        parent.closest('code') ||
                        parent.closest('.glossary-term')) {
                        return NodeFilter.FILTER_REJECT;
                    }
                    return NodeFilter.FILTER_ACCEPT;
                }
            }
        );
        
        let node;
        while (node = walker.nextNode()) {
            if (node.textContent.trim().length > 0) {
                textNodes.push(node);
            }
        }
        
        textNodes.forEach(textNode => {
            const text = textNode.textContent;
            const parent = textNode.parentElement;
            
            // Check each glossary term
            Object.keys(glossary).forEach(term => {
                const regex = new RegExp(`\\b${term}\\b`, 'gi');
                if (regex.test(text)) {
                    // Split text and wrap matching terms
                    const parts = text.split(regex);
                    const matches = text.match(regex);
                    
                    if (matches && matches.length > 0) {
                        const fragment = document.createDocumentFragment();
                        
                        parts.forEach((part, index) => {
                            if (part) {
                                fragment.appendChild(document.createTextNode(part));
                            }
                            
                            if (index < matches.length) {
                                const span = document.createElement('span');
                                span.className = 'glossary-term';
                                span.textContent = matches[index];
                                span.style.cssText = `
                                    color: var(--md-primary-fg-color);
                                    cursor: help;
                                    border-bottom: 1px dotted var(--md-primary-fg-color);
                                    text-decoration: none;
                                    transition: background-color 0.2s;
                                `;
                                
                                span.addEventListener('mouseenter', function(e) {
                                    this.style.backgroundColor = 'var(--md-default-fg-color--lightest)';
                                    showTooltip(term, glossary[term.toLowerCase()], this);
                                    currentTerm = term;
                                });
                                
                                span.addEventListener('mouseleave', function() {
                                    this.style.backgroundColor = 'transparent';
                                    hideTooltip();
                                });
                                
                                fragment.appendChild(span);
                            }
                        });
                        
                        parent.replaceChild(fragment, textNode);
                    }
                }
            });
        });
    }
    
    // Initialize when page loads
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', findGlossaryTerms);
    } else {
        // Small delay to ensure content is fully rendered
        setTimeout(findGlossaryTerms, 100);
    }
    
    // Hide tooltip when scrolling
    window.addEventListener('scroll', hideTooltip, true);
})();
