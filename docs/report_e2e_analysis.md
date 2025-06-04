# Report: Layerwise Analysis of HuBERT vs Wav2Vec2.0

1. **Layer Functionality Progression**:
   - Both HuBERT and Wav2Vec have three distinct clusters (layers 0-1, 2-6, and 7-12) which aligns with how lower layers capture acoustic features, middle layers handle contextual integration, and higher layers develop more sophisticated representations.

2. **Early Layer Function**:
   - Earliest layers (0-1) in both models process basic acoustic features, forming the foundation for higher-level representations.

3. **Contextual Information**:
   - The middle layers (2-6) in both models seem to be responsible for contextual integration. This is how the models begin to incorporate context into their representations. 
   
## Additional thoughts

1. **Variance Patterns**:
   - HuBERT shows wider variance (up to 20.0) compared to Wav2Vec 2.0's narrower range (under 3.5) suggesting that HuBERT has more feature diversity within individual layers.
   - The peak variance location difference (layer 12 for HuBERT vs. layer 11 for Wav2Vec 2.0) indicates a fundamental architectural difference between both architectures.
   - Variance is important because it indicates that a layer encodes more information (more variation within the data).(?)

2. **Layer Transitions**:
   - HuBERT is observed having smoother transitions between layers versus Wav2Vec 2.0's more abrupt boundaries. This can provide insight into how information flows/gets transformed across layers of the networks.

3. **Feature Distribution Differences**:
   - HuBERT's more dispersed distribution with extreme outliers versus Wav2Vec 2.0's more structured, curved pattern suggests different approaches to feature representation for each model. 
   - The presence of extreme outliers in HuBERT suggest is allocates significant capacity to specific speech characteristics (??) potentially that warrant distinctive encoding.

## Key Insights Not Covered in My Previous Response

1. **PCA Explained Variance Pattern**: (loosley explained)
   - The decreasing explained variance in HuBERT's later layers versus higher explained variance in Wav2Vec 2.0's layer 12 suggests different information compression strategies. More explictily:
        - When PCA explained variance is lower (as in HuBERT's later layers dropping to ~15%), it suggests the model is distributing information more evenly across many dimensions rather than concentrating it in a few dominant components. This means the representation is using more of its available dimensions to encode information, potentially capturing more subtle distinctions or relationships. On the other hand, it makes interpretation more challenging, as important information isn't concentrated in a few easily analyzable dimensions. 

        - HuBERT's decreasing explained variance in later layers suggests these representations become increasingly complex and distributed as they progress through the network.

        - Robusntess - when variance is distributed across many components, it suggests the model isn't relying heavily on any single feature or small set of features to make distinctions. This could indicate greater robustness, as the model doesn't depend critically on a few key features that might be disrupted by noise or domain shifts.

        - Wav2Vec 2.0 maintains higher explained variance in layer 12 (22%) compared to earlier layers which may suggest it retains a more concentrated representation even in its final layers.
        (?)This could reflect Wav2Vec 2.0's contrastive objective, which might benefit from more clearly separated, distinctive features rather than widely distributed representations.

2. **t-SNE Clustering Differences**:
   - HuBERT forming distinct clusters in later layers (e.g.,layer 12) versus Wav2Vec 2.0 showing more structured trajectory-like patterns (elongated/curved paths rather than disticnt, separated groups) suggests fundamentally different ways of organizing speech representations. So why does HuBERT form more distinct clusters?
        - HuBERT has a masked prediction objective - predicts discrete speech units for masked segments and thus encourages the formation of more categorical representations. Such discrete units (like phonemes or acoustic untis) enables the models to create more clear decision boundaries between different speech sounds. As a result this leads to more cluster-like structures. 
        - "Cluster-then-predict" - HuBERT uses a k-means clustering process to discover its own targets in an iterative and self-reinforcing loop: (steps 1 and 2 essentially incorporate clustering behaviour directly into the learning objective)
                1. Initial clustering creates discrete targets (cluster)
                2. Model learns to predict these clusters (predict)
                3. Representations become increasingly organized around these discovered clusters
        - BERT-like architecture helps create representations that correspond to meaningful linguitic categories rather than just acoustic similarities. 

        - Contrastimg with Wav2Vec2.0


The findings support the conclusion that while both models process speech in hierarchical fashion, HuBERT shows more gradual refinement while Wav2Vec 2.0 exhibits more discrete functional specialization between layer groups.