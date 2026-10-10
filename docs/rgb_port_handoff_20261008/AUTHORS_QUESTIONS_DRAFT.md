# Borrador opcional — NO enviado

Subject: Garl-TTC / EvTTC evaluation contract and portable FCWD calibration

Dear Garl-TTC and Event-Aided TTC authors,

Thank you for the released data, code and models. We are preserving the distinction between our exploratory transfer evaluation and a reproduction of your official results. Could you clarify:

1. Which checkpoint hash, source revision and preprocessing configuration produced Table VI? What training/pretraining/selection data did those checkpoints use, including the public TRAIN40 versus training-listed46 distinction?
2. Which recording IDs, overlap/run variants, timestamps and camera/lens correspond to the table's short sequence names? We need original recording/clip mappings, not to rename already observed clips as independent data.
3. What exact ROI availability, crop, event windows, RGB pairing, target anchor/alignment, valid-range, failure handling and aggregation define the evaluation? A runnable scorer would be particularly helpful.
4. Could you export the FCWD MATLAB MCOS calibration to a portable structure with intrinsics, distortion, camera IDs, resolutions, extrinsics/rectification maps and coordinate conventions, and identify the provenance of innercar_bbox.csv? We do not want to infer an event-to-RGB mapping from image sizes or use TTC labels to fit it.

Best regards,
Álvaro

Do not send this message without a separate explicit instruction. Lack of a reply does not block the new eAP RGB training.
