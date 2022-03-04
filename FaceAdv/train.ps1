Param(
    [String]$model=$(throw "Parameter missing: -model 'FaceNet', 'ArcFace' or 'CosFace'"),
    [String]$a_mode=$(throw "Parameter missing: -a_mode 'target or 'untarget'"),
    [int32]$class_start=$(throw "Parameter missing: -class_start 0"),
    [int32]$class_end=$(throw "Parameter missing: -class_end"),
    [int32]$batch_size=$(throw "Parameter missing: -batch_size")
)


# changing workspace
conda activate pytorch
# python face_adv.py --mode=train --attacked_model=ArcFace --epochs=1 --attack_mode=target --target_class=11 --batch_size=16 --load_pretrained
$target_class = $class_start
$mode = "train"
for (; $target_class -le $class_end; $target_class++){
    # Summary
    $summary = "Target Model [{0}], Target Class [{1}], Batch Size [{2}], Attack Mode [{3}]" -f $model, $target_class, $batch_size, $a_mode
    Write-Output $summary

    # praparing params
    $arg1 = "--mode={0}" -f $mode
    $arg2 = "--attacked_model={0}" -f $model
    $arg3 = "--epochs=1"
    $arg4 = "--attack_mode={0}" -f $a_mode
    $arg5 = "--target_class={0}" -f $target_class
    $arg6 = "--batch_size={0}" -f $batch_size
    $arg7 = "--load_pretrained"

    python face_adv.py $arg1, $arg2, $arg3, $arg4, $arg5, $arg6, $arg7
}
