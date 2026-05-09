**Synthetic Healthcare Data Generation Agent**

This LLM is connected to a conditional variational autoencoder (cVAE) that I made in order to generate synthetic patient data from natural language prompts within an agent session. Alternatively, the API can be used directly without the LLM agent intermediary.

Uses Anthropic API for LLM agent part, but it can also be integrated with an on-prem model by tweaking the code to connect to that model instead (through a different library than Anthropic).

The system prompt is designed with many prompt engineering strategies in order to ensure the tool call is called appropriately and immediately without giving the agent room to hallucinate.

The cVAE shown uses sliced wasserstein distance instead of KL divergence, which showed an improvement in performance. See api.py for more details on how the model works. It uses the pkl and pt files for scalers and columns and weights of the models from my training, respectively, to run the model here. These are ommitted from the repository because it is derived from non-public.

The cVAE was part of a bigger project that compares many models that generate patient data trained on the [MIMIC-III dataset](https://physionet.org/content/mimiciii/1.4/). To see the reports made and final report and presentations along with the code of the 2 cVAE variations tested (other models from project may not be directly shown), see [its GitHub repository](https://github.com/StevePatpatyan/Synthetic-Healthcare-Data-Generation-CVAE).

No explicit license stated at the moment.

Data and Legal Limitations
•	The model is trained on MIMIC-III, which is credentialed data. The trained weights cannot be openly distributed, limiting the system’s shareability.
•	The system generates de-identified synthetic data but has not been formally evaluated for re-identification risk or clinical safety.
