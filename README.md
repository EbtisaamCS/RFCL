<html>
<head>
    <title>Robust Federated Learning Method against Data and Model Poisoning Attacks with Heterogeneous Data Distribution</title>
</head>
<body>
<h2>Overview</h2>
    <p>RFCL is an implementation of a novel federated learning approach that combines multi-centre meta-learning, 
      internal robust aggregation, cosine similarity-based cluster selection, and personalized model sharing to enhance the robustness 
      and performance of federated learning systems. This repository contains the source code and resources to reproduce the results presented 
      in the paper "Robust Federated Learning Method against Data and Model Poisoning Attacks with Heterogeneous Data Distribution". The paper has been accepted to the ECAI2023 </p>
   <img src="rfcl_logo.png" alt="Overview of RFCL">
    <h2>Features</h2>
    <ul>
        <li>Multi-center meta-learning for generating representative cluster centres </li>
        <li>Internal robust aggregation for fair and effective model aggregation</li>
        <li>Cosine similarity-based cluster selection for improved external aggregation</li>
        <li>Personalized model sharing to align models with specific data distributions</li>
        <li>Robustness against poisoning attacks with non-IID data</li>
    </ul>
    <h2>Robust Aggregation in FL</h2>
    <p>We Have six main aggregation schemes, including FedAvg (not robust), MKrum, Median, AFA, FedMGDA+, and CC. These aggregation methods are 
    carefully evaluated to assess their robustness against different adversarial attacks and data distribution scenarios.</p>

    <h2>FL Adversarial Strategies</h2>
    <p>We explore different adversarial attack strategies, such as Inner Product Manipulation (IPM), A Little Is Enough (ALIE), sign-flipping, random noise injection, and label-flipping. The evaluation considers various numbers of attackers and Non-IID data distributions on multiple datasets, including MNIST, CIFAR-10, and Fashion-MNIST.</p>

    <h2>Work built upon</h2>
    <p>This project is built upon the work of Samuel Trew's Federated Learning repository, available at <a href="https://github.com/SamuelTrew/FederatedLearning">https://github.com/SamuelTrew/FederatedLearning</a>.</p>


</body>
</html>
