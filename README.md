# Efficient Transformer architecture for high-dynamic image acquisition on lightweight systems.
Authors: Steven Tel, Barthelemy Heyrman, Dominique Ginhac

## Presentation

High dynamic range (HDR) imaging remains a challenge for modern digital photography. Recent research proposed high-quality HDR acquisition solutions, but at the cost of a large number of operations and a long inference time, making it difficult to implement these solutions on lightweight systems. We propose a new efficient Transformer architecture for HDR imaging based on an additive attention module. To our knowledge, our solution is the first Transformer architecture for HDR imaging that can
be executed on lightweight systems. By performing qualitative and quantitative comparisons of our network with the state of the art, we demonstrate that our network produces competitive results in terms of quality while being faster than the state of the art.
Experimental results show our method obtains a µ-PSNR score of 44.13 on the reference dataset proposed by Kalantari et al. and can be executed at 11 frames per second using a Apple M1 neural processor.
