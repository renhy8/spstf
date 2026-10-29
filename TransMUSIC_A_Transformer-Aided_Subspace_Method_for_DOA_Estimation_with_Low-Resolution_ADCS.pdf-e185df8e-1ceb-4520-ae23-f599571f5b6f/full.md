# TRANSMUSIC: A TRANSFORMER-AIDED SUBSPACE METHOD FOR DOA ESTIMATION WITH LOW-RESOLUTION ADCS

Junkai Ji, Wei Mao, Feng Xi, Shengyao Chen

## ABSTRACT

Direction of arrival (DOA) estimation employing low-resolution analog-to-digital convertors (ADCs) has emerged as a challenging and intriguing problem, particularly with the rise in popularity of large-scale arrays. The substantial quantization distortion complicates the extraction of signal and noise subspaces from the quantized data. To address this issue, this paper introduces a novel approach that leverages the Transformer model to aid the subspace estimation. In this model, multiple snapshots are processed in parallel, enabling the capture of global correlations that span them. The learned subspace empowers us to construct the MUSIC spectrum and perform gridless DOA estimation using a neural network-based peak finder. Additionally, the acquired subspace encodes the vital information of model order, allowing us to determine the exact number of sources. These integrated components form a unified algorithmic framework referred to as TransMUSIC. Numerical results demonstrate the superiority of the TransMUSIC algorithm, even when dealing with one-bit quantized data. The results highlight the potential of Transformer-based techniques in DOA estimation.

Index Terms— DOA estimation, Transformer, low-resolution ADCs, quantization

## 1. INTRODUCTION

The task of direction of arrival (DOA) estimation plays an essential role in a wide range of applications, including radar and sonar positioning, wireless communication, sensor networks, and so on [1,2]. A variety of algorithms have emerged to deduce DOAs from single or multiple snapshots of received signals. These encompass classical subspace-based techniques [3–6], as well as compressed sensing (CS)-based methods [7–10]. Notably, while each snapshot denotes a discrete-time sample of the received signal, conventional DOA estimation methods often overlook the influence of quantization, which discretizes signal amplitude. Most current array systems resort to high-resolution ADCs to minimize the induced quantization distortion. However, this practice comes at the cost of increased expenses, power consumption, and substantial data volume, particularly concerning large-scale arrays. Therefore, it poses a significant challenge to develop an efficient DOA estimation method using low-resolution ADCs, which explicitly takes account of the effect of quantization distortion.

In the context of DOA estimation using low-resolution ADCs, the significant quantization distortion complicates the feasibility of employing subspace-based methods. While techniques like the arcsin law [11], also known as the Bussgang theorem [12], can be used to reconstruct the unquantized normalized covariance matrix from one-bit quantized data, thereby facilitating the use of subspace-based methods as demonstrated in [13–16], a large performance gap exists between these one-bit DOA estimation methods and their classic counterparts. Furthermore, using the time-varying sampling thresholds will significantly complicate the deriving of the covariance matrix from one-bit quantized data [17]. Moreover, it is challenging to extend the Bussgang theorem to cases involving arbitrary low-bit quantizers. Another approach to perform DOA estimation with low-resolution ADCs relies on quantized CS theory [18,19], which exploits the inherent sparsity of the signals and recovers the sparse solutions through convex optimization [20]. While the covariance fitting techniques [21–24] partially address the intrinsic off-grid issue associated with CS-based methods, the surge in computational complexity imposes limitations on their practical applications.

The recent development of deep learning has paved the way for the emergence of diverse data-driven DOA estimation methods [25–28]. These approaches leverage multi-layer neural networks (NNs) to learn the intricate relationship between DOAs and the received data or the covariance matrix by training on a given data set. While such blackbox neural networks enable the approximation of complex nonlinear mappings, they do so without utilizing the underlying knowledge of the DOA model, lacking the interpretability of those classic modelbased methods. These models are typically trained on massive datasets, limiting their adaptability when confronted with different scenarios.

Building upon the paradigm of model-based deep learning [29,30], recent works [31–34] have introduced a hybrid approach that combines both model-based and data-driven approaches for DOA estimation. The strength of this hybrid approach lies in its ability to deal with the model mismatch problem by resorting to data-driven techniques, while simultaneously exploiting the model knowledge to simplify the NN-based architecture. For example, the DA-MUSIC algorithm [31, 33] first employs a recurrent neural network (RNN) to learn a surrogate covariance matrix from the received data, then seamlessly integrates this learned information with the established structure of the classic MUSIC algorithm to estimate the DOAs. This combination empowers the datadriven aspect to successfully overcome the model mismatch issues that might arise when the actual signal model deviates from the assumed one. It is worth noting that while these hybrid approaches present a novel pathway for developing DOA estimation methods, they mainly focus on constructing data-driven models to learn so-called covariance matrices, which are then used to estimate subspaces via eigenvalue decomposition (EVD). Thus, subspace estimation still heavily relies on model-based methods, which is a computationally intensive task.

This paper is dedicated to developing a hybrid approach to recover high-resolution DOAs from the data involving significant quantization distortion. Specifically, we use a Transformer module [35] to directly learn the noise subspace from the quantized data, by exploiting the attention mechanism to capture underlying correlations present in the data. In contrast to the convolutional neural network (CNN) or RNNbased architectures, the Transformer model processes multiple snapshots in parallel, enabling to capture long-range information among them. We combine the Transformer-based subspace estimator with the algorithmic streamline of the classic MUSIC algorithm and yield a comprehensive framework, referred to as TransMUSIC. By leveraging the estimated subspace, we can construct the MUSIC spectrum, facilitating the localization of DOAs by finding its peaks. Furthermore, the learned subspace, in essence, encapsulates valuable information regarding the number of sources, which in turn enables us to deduce the source numbers from it. Therefore, TransMUSIC establishes a unified framework capable of not only recovering DOAs but also estimating the model order, thereby eliminating the need for prior knowledge of the exact number of sources. Numerical results show that even with just one-bit quantized data, TransMUSIC can outperform the classic MUSIC algorithm utilizing unquantized data. This remarkable performance proves the effectiveness of the attention mechanism in extracting the critical features for accurate DOA estimation.

## 2. SIGNAL MODEL AND MOTIVATIONS

In this section, we first describe the array signal model and the quantization operation, then briefly introduce the classic MUSIC algorithm, and finally discuss the motivation of this work.

## 2.1. Signal Model

Consider a scenario with K narrowband far-field source signals impinging on a linear array with M omnidirectional sensors from different directions ${ \theta _ { k } } \in \left[ { - 9 0 ^ { \circ } , 9 0 ^ { \circ } } \right]$ . The locations of receive antennas are given by $\{ r _ { 1 } d , r _ { 2 } d , \cdot \cdot \cdot , r _ { M } d \}$ , where $r _ { 1 } < r _ { 2 } < \dots < r _ { M } , r _ { m } \in \mathbb { Z }$ and $\begin{array} { r } { d = \frac { \lambda } { 2 } } \end{array}$ with λ denoting the wavelength of sources. For simplicity, we assume a uniform linear array (ULA) used in this paper, i.e., $r _ { m } = m - 1$ . Then the L snapshots received at the array can be represented by

$$
\begin{array}{l} \mathbf {y} (t) = \sum_ {k = 1} ^ {K} \mathbf {a} (\theta_ {k}) x _ {k} (t) + \mathbf {n} (t) \\ \qquad = \mathbf {A} (\boldsymbol {\theta}) \mathbf {x} (t) + \mathbf {n} (t), \quad t = 1, 2, \dots , L \end{array}\tag{1}
$$

where $\mathbf { y } ( t ) , \mathbf { n } ( t ) \in \mathbb { C } ^ { M }$ , and $\mathbf { x } ( t ) \in \mathbb { C } ^ { K }$ denote the received signals, additive noises, and source signals, respectively. $\begin{array} { r l } { \mathbf { A } ( \pmb { \theta } ) } & { { } = } \end{array}$ $[ \mathbf { a } ( \theta _ { 1 } ) , \mathbf { a } ( \theta _ { 2 } ) , \cdot \cdot \cdot , \mathbf { a } ( \theta _ { K } ) ] \in \mathbb { C } ^ { M \times K }$ is the steering matrix with $\theta =$ $\{ \theta _ { 1 } , \theta _ { 2 } , \cdot \cdot \cdot , \theta _ { k } \} \mathrm { a n d } \mathbf { a } ( \theta _ { k } ) = [ e ^ { j 2 \pi \frac { r _ { 1 } d \sin \theta _ { k } } { \lambda } } , \cdot \cdot \cdot , e ^ { j 2 \pi \frac { r _ { M } d \sin \theta _ { k } } { \lambda } } ] .$

Before performing digital processing, the received signal y(t) has to be converted to a set of digital representations by using $\bf { A D C s } .$ Classic DOA estimation methods typically assume the use of highresolution ADCs, effectively disregarding the influence of quantization distortion. However, in this paper, we explicitly consider the effect of quantization distortion and apply only 1-bit quantization to the observed signal. Specifically, we use a pair of one-bit ADCs to independently quantize the real and imaginary components of the received signal. Let $\begin{array} { r }  \dot { \mathcal { Q } } _ { 1 } ( \cdot ) = \frac { 1 } { \sqrt { 2 } } ( \mathrm { s i g n } ( \Re \{ \cdot \} \dot { \} ) + j \mathrm { s i g n } ( \Im \{ \cdot \} ) ) } \end{array}$ be the complex one-bit quantization operator, where sign(·) denote the sign function applied element-wise to any vector or matrix. Thus, the one-bit quantized data becomes

$$
\mathbf {z} (t) = \mathcal {Q} _ {1} (\mathbf {y} (t)) = \mathcal {Q} _ {1} (\mathbf {A} (\boldsymbol {\theta}) \mathbf {x} (t) + \mathbf {n} (t)).\tag{2}
$$

Therefore, the DOA estimation is to recover the set of angles θ from the quantized data $\{ { \mathbf { z } } ( t ) \} _ { t = 1 } ^ { L }$ . Due to the low-resolution $\mathbf { A D C s }$ we have to explicitly take the distortion induced by quantization into account and design the specific algorithm for DOA estimation.

## 2.2. SubSpace-based DOA Estimation

Subspace methods are a class of DOA estimation algorithms that hinge on precisely estimating both signal and noise subspaces derived from the received signals. A representative algorithm within this category is the MUSIC algorithm [3], which deduces the subspaces and recovers the DOAs from the covariance matrix $\mathbf { R } _ { \boldsymbol { y } } \triangleq \mathbb { E } \{ \mathbf { \boldsymbol { y } } ( t ) \mathbf { \boldsymbol { y } } ^ { H } ( t ) \}$ }. The inferences in the MUSIC algorithm are based on the following assumptions:

A1 The number of sources is known, and the source signals are uncorrelated, i.e., $\mathbf { x } ( t ) \sim \mathcal { C N } ( \mathbf { 0 } , \mathbf { P } )$ , where $\mathbf { P } = \mathrm { d i a g } ( p _ { 1 } , p _ { 2 } , \cdot \cdot \cdot$ • $p _ { K } )$ denotes the covariance matrix of ${ \bf x } ( t ) ;$

A2 The additive noise are independently and identically distributed as $\mathcal { C N } ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } _ { M } )$ , where $\sigma ^ { 2 }$ is the variance of noise;

A3 There is no temporal correlation between each snapshot, and the additive noise is uncorrelated with the sources.

Then the noise and signal subspaces can be identified via EVD of the covariance matrix $\mathbf { R } _ { y } ^ { - } = \mathbf { A } ( \pmb { \theta } ) \mathbf { P } \mathbf { A } ^ { H } ( \pmb { \theta } ) + \sigma ^ { 2 } \mathbf { I } _ { M }$ . By restricting the number of sources to be less than the number of antennas, i.e., $K < M .$ the eigenvectors corresponding to the largest K eigenvalues consist of the signal subspace, while the eigenvectors corresponding to the remaining $M - K$ eigenvalues consists of the noise subspace. We can construct the following MUSIC spectrum:

$$
P _ {\mathrm{MUSIC}} (\psi) = \frac {1}{\mathbf {a} ^ {H} (\psi) \mathbf {E} _ {n} \mathbf {E} _ {n} ^ {H} \mathbf {a} (\psi)},\tag{3}
$$

where $\mathbf { E } _ { n } \ \in \ \mathbb { C } ^ { M \times ( M - K ) }$ denotes the noise subspace. By exploiting the orthogonality of signal and noise subspaces, the DOAs are estimated by finding the K peaks of $P _ { \mathrm { M U S I C } } ( \psi )$ . In practical scenarios, finite-length snapshots necessitate the use of the sample covariance matrix (SCM) $\begin{array} { r } { \hat { \mathbf { R } } y = \frac { 1 } { L } \sum _ { t = 1 } ^ { L } \mathbf { y } ( t ) \mathbf { y } ^ { H } ( t ) } \end{array}$ as a substitute for $\mathbf { R } _ { y } .$

Nonetheless, when using the low-resolution $\mathbf { A D C s } ,$ the data available is the quantized observations $\{ { \mathbf { z } } ( t ) \} _ { t = 1 } ^ { L } .$ rather than the received signal $\{ \mathbf { y } ( t ) \} _ { t = 1 } ^ { L } .$ While it is possible to establish the relation between $\mathbf { R } _ { y }$ and $\mathbf { R } _ { z } \triangleq \{ \mathbf { z } ( t ) \mathbf { z } ^ { H } ( t ) \}$ using the arcsine law [11] or the Bussgang theorem [12], the performance of subspace estimation is significantly degraded due to the intricate interplay between the quantized measurements and the desired signal and noise subspaces. Consequently, the task of accurately deducing subspaces from quantized observations becomes a notable challenge, owing to its pivotal role in achieving precise high-resolution DOA estimation.

## 2.3. Motivations

A vital limitation of the classic MUSIC algorithm lies in its incapability to deduce the subspaces when the assumptions $\mathbf { A } 1 { \sim } \mathbf { A } 3$ are violated. In cases where the number of sources is unknown, additional processing is required to determine the exact number of sources present in the signals [36, 37]. Furthermore, quantization distortion impairs the covariance matrix’s ability to capture the information of signal and noise subspaces precisely. In practical scenarios, the SCM deviates significantly from the actual covariance matrix due to limited snapshots, further increasing inaccuracies in EVD-based subspace estimation. These challenges motivate us to develop a data-driven method to establish a robust nonlinear mapping between the quantized data and the desired signal or noise subspace.

## 3. TRANSMUSIC

In this section, we present the architecture of the proposed TransMU-SIC, which builds upon the algorithmic framework of the classic MU-SIC algorithm and replaces the EVD-based subspace estimation with a Transformer encoder. The overall architecture of the TransMUSIC algorithm is shown in Fig.1, consisting of a subspace estimator, a DOA estimator, and a source number (SN) estimator. In particular, the transformer encoder processes the L snapshots in parallel to extract essential features of the quantized data. These features contribute to subspace estimation through a dedicated fully connected network (FCN). Then we can construct the MUSIC spectrum (3) and estimate the DOAs facilitated by a peak-finding network based on another FCN. A significant advantage of the TransMUSIC is its capability to perform subspace estimation without prior knowledge of the number of sources. Remarkably, the information about the source number is implicitly embedded within the estimated subspace, which can be extracted using an additional FCN branch. We will elaborate on the architecture in the following subsection.

## 3.1. Architecture

In the TransMUSIC, the transformer encoder employs the attention mechanism for effective feature extraction by concurrently processing the quantized observations $\{ { \mathbf { z } } ( t ) \} _ { t = 1 } ^ { L }$ . To facilitate the application of the attention mechanism, we concatenate the real and imaginary parts of each snapshot z(t) into a vector $\hat { { \mathbf { z } } } ( t ) = [ \Re ( { \mathbf { z } } ( t ) ) ^ { T } , \breve { \Im } ( { \mathbf { z } } ( \dot { t } ) ) ^ { T } ] ^ { T }$ which then serves as the input to the transformer encoder. Typically, the transformer encoder comprises a sequence of interconnected blocks, each of which encompasses a multi-head attention module, followed by a one- or two-layer FCN with an activation function (e.g., ReLU). The output of the transformer encoder constitutes a collection of features, denoted as $\{ \mathbf { o } ( t ) \} _ { t = 1 } ^ { L } ,$ and these features maintain the same dimensionality as the input data.

![](images/b0e4ad21f88651f5dd0af641a62415d0df76af7332a075670bb3b2157407a8e5.jpg)  
Fig. 1: The architecture of the TransMUSIC algorithm.

Subsequently, these features are leveraged to formulate an estimate of the noise subspace, a crucial step in generating the MUSIC spectrum. It is notable that in the ULA array with M antennas, the covariance matrix $\mathbf { R } _ { y }$ is a Hermitian and Toeplitz matrix. This unique property allows for a parametric representation of the covariance matrix through an Mdimensional vector. Based on this insight, an average unit is employed to compactly represent the extracted features into one feature vector $\begin{array} { r } { \bar { \mathbf { o } } = \frac { 1 } { L } \sum _ { t = 1 } ^ { L } \mathbf { o } ( t ) } \end{array}$ , facilitating subsequent processing independently of the input data’s size. Following this, an FCN network is employed to generate the desired noise subspace based on the feature vector. Considering the dimension of the actual noise subspace $\mathbf { E } _ { n }$ depends on the number of sources $K$ which is both varying and unknown, the FCN network outputs an augmented noise subspace $\tilde { \mathbf { E } } _ { n } \in \mathbb { C } ^ { M \times \dot { M } }$ in its place. Essentially, $\tilde { \mathbf { E } } _ { n }$ can be viewed as a matrix with $M - K$ columns being the dominant ones.

After formulating the MUSIC spectrum (3) using the augmented noise subspace $\tilde { \mathbf { E } } _ { n }$ , we use an additional FCN network to obtain the DOAs by finding the peaks of the spectrum. The application of an FCNbased peak finder avoids the exhaustive search on the discrete grids, achieving superior resolution. Addressing the dynamic output dimension of the FCN network, especially in diverse scenarios with varying numbers of sources, we resolve this by setting the output dimension to M − 1. This choice aligns with the maximum number of sources a ULA with M antennas can effectively resolve. However, only the first K outputs are considered as the estimated DOAs while the remaining outputs are ignored. The derivation of K will be addressed by the SN estimator.

The learned augmented noise subspace $\tilde { \mathbf { E } } _ { r }$ <sub>n</sub> inherently encapsulates the information about K, facilitating the determination of the number of sources. For this purpose, we construct an SN estimator comprising an FCN network followed by a softmax layer, as illustrated in Fig. 1. Given that the maximum resolvable source number is M −1, the task is to resolve M − 1 classifications. This naturally translates into a multiclass classification problem that can be effectively solved by using an FCN network.

## 3.2. Training Procedure

The TransMUSIC algorithm undergoes end-to-end supervised training. The training set D consists of D pairs of quantized observations and their corresponding DOAs, i.e., $\begin{array} { r } { \hat { \mathcal { D } } = \left\{ \left( \mathbf { \tilde { Z } } ^ { ( d ) } , \pmb { \theta } ^ { ( d ) } \right) \right\} _ { d = } ^ { D } } \end{array}$ with 1 $\mathbf { Z } ^ { ( d ) } = [ \mathbf { z } ^ { ( d ) } ( 1 ) , \cdots , \mathbf { z } ^ { ( d ) } ( L _ { d } ) ] \in \mathbb { C } ^ { M \times L _ { d } }$ and $| \pmb { \theta } ^ { ( d ) } | = K _ { d }$ . Let $\hat { \pmb { \theta } } ( \mathbf { Z } ^ { ( d ) } )$ and $\hat { \mathbf { K } } ( \tilde { \mathbf { E } } _ { N } ^ { ( d ) } )$ denote the $( M - 1 )$ -dimension outputs of the DOA estimator and the SN estimator, respectively, applying to the data $\mathbf { Z } ^ { ( d ) }$ and the learned noise subspace $\tilde { \mathbf { E } } _ { N } ^ { ( d ) }$ . To train the lower branch in Fig. 1 for DOA estimation, we employ the root mean squared periodic error (RMSPE) [38, 39] as the loss function:

$$
\mathcal {L} _ {\mathrm{RMSPE}} (\boldsymbol {\theta} ^ {(d)}, \hat {\boldsymbol {\theta}} (\mathbf {Z} ^ {(d)})) = \mathrm{RMSPE} (\boldsymbol {\theta} ^ {(d)}, \hat {\boldsymbol {\theta}} _ {1: K _ {d}} (\mathbf {Z} ^ {(d)})),\tag{4}
$$

with RMSPE(θ, θ<sup>ˆ</sup>) = min $\mathop { \bf \cdot } \mathop { \bf \in } \mathcal { P } _ { K } \left( \frac { 1 } { K } \right|$ mod<sub>π</sub> $( \pmb \theta - \mathbf P \hat { \pmb \theta } ) \| ^ { 2 } ) ^ { 1 / 2 }$ , where P is the set of all $K \times K$ permutations and mod<sub>π</sub> denotes the elementwise modulus operation. The SN estimator is trained using the crossentropy as the loss function:

$$
\mathcal {L} _ {\mathrm{CE}} (K _ {d}, \hat {\mathbf {K}} (\tilde {\mathbf {E}} _ {N} ^ {(d)})) = - \log \hat {K} _ {K _ {d}} (\tilde {\mathbf {E}} _ {N} ^ {(d)}),\tag{5}
$$

where $\hat { K } _ { K _ { d } } ( \tilde { \mathbf { E } } _ { N } ^ { ( d ) } )$ denotes the $K _ { d }$ -th element of $\hat { \mathbf { K } } ( \tilde { \mathbf { E } } _ { N } ^ { ( d ) } )$ . It is worth noting that the gradient propagation from the SN estimator to the subspace estimator should be blocked during backpropagation.

## 3.3. Discussion

Similar to the DA-MUSIC [31,33], the TransMUSIC we present here is built upon the algorithmic framework of the classic MUSIC algorithm. A key distinction between them lies in our utilization of a transformer encoder, which plays a central role in capturing the intricate nonlinear relationship between the quantized data and the noise subspace. It performs this task through parallel processing, in contrast to the RNN in the DA-MUSIC algorithm, which sequentially analyzes the dependencies between consecutive snapshots.

Another notable advantage of TransMUSIC is that it directly estimates the noise subspace from the received data, circumventing the need for the computationally demanding EVD involved in subspace estimation from the covariance matrix. This improvement significantly accelerates the overall DOA estimation procedure. Moreover, due to its direct subspace estimation approach, TransMUSIC eliminates the necessity of using a selector to choose the corresponding eigenvectors, as done in the DA-MUSIC algorithm. This simplifies the architecture and the training procedure.

While the motivation of our work is to develop an NN-aided DOA estimation method addressing the effect of quantization distortion, it is worth noting that TransMUSIC’s applicability is not confined to such scenarios. The power of the transformer module facilitates the extension of our work to other non-ideal cases, such as coherent sources, arbitrary array geometry, and scenarios with some malfunctioning antennas. We will leave these for future work.

## 4. NUMERICAL EXPERIMENTS

Now we evaluate the performance of the proposed TransMUSIC algorithm<sup>1</sup>. Throughout the experiments, we consider a ULA with $M = 8$ half-wavelength spaced elements. The source signal x(t) and noise signal n(t) are both modeled as complex Gaussian distributions, in accordance with the assumptions A1∼A3 in Subsection 2.2. The DOAs θ are randomly drawn from an i.i.d. uniform distribution over the range $\left( - { \frac { \pi } { 2 } } , { \frac { \pi } { 2 } } \right)$ and the number of sources is randomly distributed between [2, 5]. To facilitate our evaluations, we create synthetic datasets through simulations in different scenarios. Unless stated otherwise, the parameters used in our experiments are summarized in Table 1.

![](images/f519e9569c6e3c2876b7fc63a8d9cd35d66843bdd9eab31c15eccf038082891f.jpg)  
Fig. 2: RMSPE with respect to various SNRs.

![](images/1bf53657c9c43d83e54d10e248f95d7f797ad13fa67f6c1e3c71be5331286cb1.jpg)  
Fig. 3: Probability of successfully estimating K versus various SNRs.

Table 1: Simulation Parameters

<table><tr><td>Parameter</td><td>Value</td><td>Parameter</td><td>Value</td></tr><tr><td>M</td><td>8</td><td>Training set size</td><td>256000</td></tr><tr><td>K</td><td>2,3,4,5</td><td>Validation set size</td><td>25600</td></tr><tr><td>L</td><td>200</td><td>Test set size</td><td>25600</td></tr><tr><td>SNR</td><td>0dB,5dB,10dB</td><td>Batch size</td><td>64</td></tr><tr><td>Optimizer</td><td>Adam</td><td>Initial learning rate</td><td> $10^{-3}$ </td></tr></table>

For the sake of comparison, we also implement several alternative approaches, including the classic MUSIC algorithm [3], the conventional beamformer method [40], and the DA-MUSIC algorithm [33]. Notably, all of these methods employ unquantized data as their input. Furthermore, we also implement the one-bit MUSIC algorithm [15] utilizing one-bit quantized data. Two TransMUSIC algorithms are implemented, training separately using one-bit quantized data and unquantized data. With the unknown number of sources, the classic MU-SIC algorithm and one-bit MUSIC algorithm determine the number of sources by estimating the multiplicity of the smallest eigenvalue while the beamformer method determines it by estimating the number of dominant peaks.

In the first experiment, we train the TransMUSIC and DA-MUSIC algorithms using the dataset mixed with SNRs of 0dB, 5dB, and 10dB, and then test the generalization of the algorithms across varying noise levels. Fig. 2 illustrates the RMSPE of different algorithms with respect to various SNRs. Both TransMUSIC algorithms, one with unquantized data and the other with one-bit data, consistently exhibit lower RM-SPEs compared to the alternative methods. Due to the diverse SNR scenarios used during training, the DA-MUSIC algorithm performs worse than the classic MUSIC algorithm, showing its limited adaptability to varying SNR conditions. However, the TransMUSIC algorithm demonstrates robust performance even when applied to SNR scenarios that it was not explicitly trained on. It is worth highlighting that TransMU-

![](images/d58eeea20e4cc10980446cd3a39ec5075bc5abb2d5e986882b27d0d5d35f3b5e.jpg)  
Fig. 4: RMSPE with respect to different number of snapshots L.

![](images/2df498217fa3ec868b2f4d66012a487ac11da1a0a417b9a75e381e741a3f7c81.jpg)  
Fig. 5: Probability of successfully estimating K versus different number of snapshots $L .$

SIC with one-bit data even outperforms those with unquantized data, underscoring the effectiveness of the Transformer architecture used for subspace estimation. We next evaluate the probability of successfully estimating the source number K versus different SNRs, depicted in Fig. 3. Again, both the TransMUSIC algorithms show better performance in identifying the exact number of sources across various SNR scenarios. Conversely, the DA-MUSIC algorithm struggled to estimate it accurately. Its accuracy notably decreased when applied to SNR scenarios outside its training scope.

In the next experiment, we maintain a fixed SNR of 10dB for training and testing. Both TransMUSIC and DA-MUSIC algorithms are exclusively trained with L = 200 snapshots and subsequently tested with varying L. Fig. 4 presents the RMSPE of various algorithms with respect to different numbers of snapshots. Although the performance degradation is observed with fewer snapshots, the TransMUSIC algorithms outperform their counterparts whether using unquantized data or using one-bit data. It shows the superior performance of the TransMU-SIC when implemented with only a few snapshots. Unfortunately, as shown in Fig. 5, the probability of estimating K significantly deteriorated with decreasing numbers of snapshots. This result highlights the dependency of source number estimation on the number of snapshots available. Addressing this shortcoming may require separate training for each number of snapshots.

## 5. CONCLUSION

In this work, we proposed a Transformer-aided subspace method, named TransMUSIC, for DOA estimation. The powerful attention mechanism used in the Transformer module makes it possible to extract valuable features for subspace estimation, even in scenarios contaminated by significant quantization distortion. TransMUSIC seamlessly integrates the Transformer-based subspace estimation into the algorithmic framework of the classic MUSIC algorithm. This not only simplifies the resulting architecture but also enhances the interpretability. Simulation results demonstrate the superior performance of the TransMUSIC algorithm, regardless of the presence of quantization distortion. Notably, the TransMUSIC algorithm, even when using onebit data, surpasses the classic MUSIC algorithm and other counterparts that rely on unquantized data. Experiments with real data will be our future work.

[1] H. Krim and M. Viberg, “Two decades of array signal processing research: the parametric approach,” IEEE Signal Processing Magazine, vol. 13, no. 4, pp. 67–94, 1996.

[2] Harry L. Van Trees, Optimum Array Processing: Part IV of Detection, Estimation, and Modulation Theory, John Wiley & Sons, Ltd, 2002.

[3] R. Schmidt, “Multiple emitter location and signal parameter estimation,” IEEE Transactions on Antennas and Propagation, vol. 34, no. 3, pp. 276– 280, 1986.

[4] P. Stoica and N. Arye, “MUSIC, maximum likelihood, and Cramer-Rao´ bound,” IEEE Trans.acoust.speech Signal Process, vol. 38, no. 5, pp. 720– 741, 1989.

[5] B.D. Rao and K.V.S. Hari, “Performance analysis of root-MUSIC,” IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 37, no. 12, pp. 1939–1949, 1989.

[6] R. Roy and T. Kailath, “ESPRIT-estimation of signal parameters via rotational invariance techniques,” IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 37, no. 7, pp. 984–995, 1989.

[7] D. Malioutov, M. Cetin, and A.S. Willsky, “A sparse signal reconstruction perspective for source localization with sensor arrays,” IEEE Transactions on Signal Processing, vol. 53, no. 8, pp. 3010–3022, 2005.

[8] Jihao Yin and Tianqi Chen, “Direction-of-arrival estimation using a sparse representation of array covariance vectors,” IEEE Transactions on Signal Processing, vol. 59, no. 9, pp. 4489–4493, 2011.

[9] Petre Stoica, Prabhu Babu, and Jian Li, “SPICE: A sparse covariancebased estimation method for array processing,” IEEE Transactions on Signal Processing, vol. 59, no. 2, pp. 629–638, 2011.

[10] Zai Yang, Lihua Xie, and Cishen Zhang, “A discretization-free sparse and parametric approach for linear array signal processing,” IEEE Transactions on Signal Processing, vol. 62, no. 19, pp. 4959–4973, 2014.

[11] J.H. Van Vleck and D. Middleton, “The spectrum of clipped noise,” Proceedings ofthe IEEE, vol. 54, no. 1, pp. 2–19, 1966.

[12] J.J. Bussgang, Crosscorrelation Functions of Amplitude-distorted Gaussian Signals, Technical Report. Research Laboratory of Electronics, Massachusetts Institute of Technology, 1952.

[13] O. Bar-Shalom and A.J. Weiss, “DOA estimation using one-bit quantized measurements,” IEEE Transactions on Aerospace and Electronic Systems, vol. 38, no. 3, pp. 868–884, 2002.

[14] Chun-Lin Liu and P. P. Vaidyanathan, “One-bit sparse array DOA estimation,” in 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017, pp. 3126–3130.

[15] Xiaodong Huang and Bin Liao, “One-bit MUSIC,” IEEE Signal Processing Letters, vol. 26, no. 7, pp. 961–965, 2019.

[16] Saeid Sedighi, Bhavani Shankar, Mojtaba Soltanalian, and Bjorn Otter-¨ sten, “One-bit DoA estimation via sparse linear arrays,” in ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2020, pp. 9135–9139.

[17] Arian Eamaz, Farhang Yeganegi, and Mojtaba Soltanalian, “Covariance recovery for one-bit sampled non-stationary signals with time-varying sampling thresholds,” IEEE Transactions on Signal Processing, vol. 70, pp. 5222–5236, 2022.

[18] Argyrios Zymnis, Stephen Boyd, and Emmanuel Candes, “Compressed sensing with quantized measurements,” IEEE Signal Processing Letters, vol. 17, no. 2, pp. 149–152, 2010.

[19] Jason N. Laska, Zaiwen Wen, Wotao Yin, and Richard G. Baraniuk, “Trust, but verify: Fast and accurate signal recovery from 1-bit compressive measurements,” IEEE Transactions on Signal Processing, vol. 59, no. 11, pp. 5289–5301, 2011.

[20] Kai Yu, Yimin D. Zhang, Ming Bao, Yu-Hen Hu, and Zhi Wang, “DOA estimation from one-bit compressed array data via joint sparse representation,” IEEE Signal Processing Letters, vol. 23, no. 9, pp. 1279–1283, 2016.

[21] Feng Xi, Yijian Xiang, Shengyao Chen, and Arye Nehorai, “Gridless parameter estimation for one-bit MIMO radar with time-varying thresholds,” IEEE Transactions on Signal Processing, vol. 68, pp. 1048–1063, 2020.

[22] Feng Xi, Yijian Xiang, Zhen Zhang, Shengyao Chen, and Arye Nehorai, “Joint angle and doppler frequency estimation for MIMO radar with onebit sampling: A maximum likelihood-based method,” IEEE Transactions on Aerospace and Electronic Systems, vol. 56, no. 6, pp. 4734–4748, 2020.

[23] Yu Xia and Feng Xi, “Grid-less DOA estimation via sparse linear array using one-bit measurements,” in 2021 CIE International Conference on Radar (Radar), 2021, pp. 2366–2370.

[24] Saeid Sedighi, M. R. Bhavani Shankar, Mojtaba Soltanalian, and Bjorn¨ Ottersten, “DoA estimation using low-resolution multi-bit sparse array measurements,” IEEE Signal Processing Letters, vol. 28, pp. 1400–1404, 2021.

[25] Ahmet M. Elbir, “DeepMUSIC: Multiple signal classification via deep learning,” IEEE Sensors Letters, vol. 4, no. 4, pp. 1–4, 2020.

[26] Zhang-Meng Liu, Chenwei Zhang, and Philip S. Yu, “Direction-of-arrival estimation based on deep neural networks with robustness to array imperfections,” IEEE Transactions on Antennas and Propagation, vol. 66, no. 12, pp. 7315–7327, 2018.

[27] Georgios K. Papageorgiou, Mathini Sellathurai, and Yonina C. Eldar, “Deep networks for direction-of-arrival estimation in low SNR,” IEEE Transactions on Signal Processing, vol. 69, pp. 3714–3729, 2021.

[28] Liuli Wu, Zhang-Meng Liu, and Zhi-Tao Huang, “Deep convolution network for direction of arrival estimation with sparse prior,” IEEE Signal Processing Letters, vol. 26, no. 11, pp. 1688–1692, 2019.

[29] Nir Shlezinger, Jay Whang, Yonina C. Eldar, and Alexandros G. Dimakis, “Model-based deep learning: Key approaches and design guidelines,” in 2021 IEEE Data Science and Learning Workshop (DSLW), 2021, pp. 1–6.

[30] Nir Shlezinger, Jay Whang, Yonina C. Eldar, and Alexandros G. Dimakis, “Model-based deep learning,” Proceedings of the IEEE, vol. 111, no. 5, pp. 465–499, 2023.

[31] Julian P. Merkofer, Guy Revach, Nir Shlezinger, and Ruud J. G. van Sloun, “Deep augmented MUSIC algorithm for data-driven DoA estimation,” in ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 3598–3602.

[32] Dor H. Shmuel, Julian P. Merkofer, Guy Revach, Ruud J. G. van Sloun, and Nir Shlezinger, “Deep root MUSIC algorithm for data-driven DoA estimation,” in ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1–5.

[33] Julian P. Merkofer, Guy Revach, Nir Shlezinger, Tirza Routtenberg, and Ruud J. G. van Sloun, “DA-MUSIC: Data-driven DoA estimation via deep augmented MUSIC algorithm,” arXiv:2109.10581, 2021.

[34] Dor H. Shmuel, Julian P. Merkofer, Guy Revach, Ruud J. G. van Sloun, and Nir Shlezinger, “SubspaceNet: Deep learning-aided subspace meth ods for DoA estimation,” arXiv:2306.02271, 2023.

[35] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, L ukasz Kaiser, and Illia Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems, 2017.

[36] Hsien-Tsai Wu, Jar-Ferr Yang, and Fwu-Kuen Chen, “Source number estimator using gerschgorin disks,” in Proceedings ofICASSP ’94. IEEE International Conference on Acoustics, Speech and Signal Processing, 1994, vol. iv, pp. IV/261–IV/264 vol.4.

[37] Zhiqiang Qiu, Ziyuan He, and Mantang Cui, “A method for signal source number estimation based on MUSIC algorithm,” in 2015 IEEE Interna tional Conference on Communication Problem-Solving (ICCP), 2015, pp. 344–346.

[38] Tirza Routtenberg and Joseph Tabrikian, “Bayesian parameter estimation using periodic cost functions,” IEEE Transactions on Signal Processing, vol. 60, no. 3, pp. 1229–1240, 2012.

[39] Tirza Routtenberg and Joseph Tabrikian, “Non-Bayesian periodic Cramer-´ Rao bound,” IEEE Transactions on Signal Processing, vol. 61, no. 4, pp. 1019–1032, 2013.

[40] M. S. Bartlett, “Smoothing periodograms from time-series with continuous spectra,” Nature, vol. 161, pp. 686–687, May 1948.