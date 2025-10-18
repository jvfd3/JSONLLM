# Datasets

## [**Table 1:** Comparison of existing AVE datasets.](https://arxiv.org/pdf/2508.11801?)

| Opensource | Multilabel | MultiDomain | Rigorous Filtering | Video Modality | Language |    Size |            Dataset |
| :--------: | :--------: | :---------: | :----------------: | :------------: | :------: | ------: | -----------------: |
|     1      |     1      |      1      |         1          |       1        | English  |  248.8K |    VideoAVE (Ours) |
|     1      |     1      |      1      |         1          |       0        | English  |  750.0K |       OA-Mine [33] |
|     1      |     1      |      1      |         1          |       0        | English  |    4.7K |       WDC-PAVE [3] |
|     1      |     1      |      1      |         0          |       0        | English  |    7.6M |           MAE [17] |
|     1      |     1      |      1      |         0          |       0        | English  |    2.2M |          MAVE [31] |
|     1      |     1      |      1      |         0          |       0        | Chinese  |   87.2K |        MEPAVE [38] |
|     1      |     0      |      1      |         0          |       0        | English  |   70.2K |   ImplicitAVE [39] |
|     1      |     0      |      1      |         0          |       0        | Chinese  |  657.4K |       AE-650K [28] |
|     0      |     1      |      1      |         0          |       0        | English  |  295.6K | e-commerce5PT [22] |
|     0      |     1      |      1      |         0          |       0        | English  |       - |       OpenTag [36] |
|     0      |     1      |      0      |         0          |       0        | English  |  333.0K |        AdaTag [29] |
|     0      |     1      |      0      |         0          |       0        | Chinese  | ~100.0K |        DESIRE [34] |

- [03] **WDC-PAVE:** Alexander Brinkmann, Nick Baumann, and Christian Bizer. 2024. Using llms for the extraction and normalization of product attribute values. arXiv e-prints (2024), arXiv–2403.
- [17] **MAE:** Robert L Logan IV, Samuel Humeau, and Sameer Singh. 2017. Multimodal attribute extraction. arXiv preprint arXiv:1711.11118 (2017).
- [22] **e-commerce5PT:** Anubhav Shrimal, Avi Jain, Kartik Mehta, and Promod Yenigalla. 2022. NERMQMRC: formulating named entity recognition as multi question machine reading comprehension. arXiv preprint arXiv:2205.05904 (2022).
- [28] **AE-650K:** Huimin Xu, Wenting Wang, Xinnian Mao, Xinyu Jiang, and Man Lan. 2019. Scaling up open tagging from tens to thousands: Comprehension empowered attribute value extraction from product title. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 5214–5223.
- [29] **AdaTag:** Jun Yan, Nasser Zalmout, Yan Liang, Christan Grant, Xiang Ren, and Xin Luna Dong. 2021. Adatag: Multi-attribute value extraction from product profiles with adaptive decoding. arXiv preprint arXiv:2106.02318 (2021).
- [31] **MAVE:** Li Yang, Qifan Wang, Zac Yu, Anand Kulkarni, Sumit Sanghai, Bin Shu, Jon Elsas, and Bhargav Kanagal. 2022. Mave: A product dataset for multi-source attribute value extraction. In Proceedings of the fifteenth ACM international conference on web search and data mining. 1256–1265.
- [33] **OA-Mine:** Xinyang Zhang, Chenwei Zhang, Xian Li, Xin Luna Dong, Jingbo Shang, Christos Faloutsos, and Jiawei Han. 2022. Oa-mine: Open-world attribute mining for e-commerce products with weak supervision. In Proceedings of the ACM Web Conference 2022. 3153–3161.
- [34] **DESIRE:** Yupeng Zhang, Shensi Wang, Peiguang Li, Guanting Dong, Sirui Wang, Yunsen Xian, Zhoujun Li, and Hongzhi Zhang. 2023. Pay attention to implicit attribute values: A multi-modal generative framework for AVE task. In Findings of the association for computational linguistics: ACL 2023. 13139–13151.
- [36] **OpenTag:** Guineng Zheng, Subhabrata Mukherjee, Xin Luna Dong, and Feifei Li. 2018. Opentag: Open attribute value extraction from product profiles. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 1049–1058.
- [38] **MEPAVE:** Tiangang Zhu, Yue Wang, Haoran Li, Youzheng Wu, Xiaodong He, and Bowen Zhou. 2020. Multimodal joint attribute prediction and value extraction for ecommerce product. arXiv preprint arXiv:2009.07162 (2020).
- [39] **ImplicitAVE:** Henry Peng Zou, Vinay Samuel, Yue Zhou, Weizhi Zhang, Liancheng Fang, Zihe Song, Philip S Yu, and Cornelia Caragea. 2024. Implicitave: An open-source dataset and multimodal llms benchmark for implicit attribute value extraction. arXiv preprint arXiv:2404.15592 (2024).

## [Table 1: Existing datasets for attribute value extraction](https://dl.acm.org/doi/pdf/10.1145/3488560.3498377)

| public | #annotations | #products | #sources |      Dataset |
| :----: | -----------: | --------: | :------: | -----------: |
|  Yes   |         2.2M |      600K |    2     |     MAE [17] |
|  Yes   |         110K |       50K |    1     | AE-110K [56] |
|  Yes   |          87K |       34K |    2     |  MEPAVE [63] |
|  Yes   |           3M |      2.2M | Multiple |         MAVE |
|   No   |         410K |      333K | Multiple |  AdaTag [59] |
|   No   |          13K |       10K |    2     | OpenTag [62] |
