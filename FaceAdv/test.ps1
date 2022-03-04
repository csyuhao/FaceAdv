Param(
    [String]$model=$(throw "Parameter missing: -model 'FaceNet', 'ArcFace' or 'CosFace'"),
    [String]$a_mode=$(throw "Parameter missing: -a_mode 'target or 'untarget'"),
    [int32]$class_start=$(throw "Parameter missing: -class_start 0"),
    [int32]$class_end=$(throw "Parameter missing: -class_end"),
    [int32]$batch_size=$(throw "Parameter missing: -batch_size"),
    [int32]$pretrained_step=$(throw "Parameter missing: -pretrained_step")
)


# changing workspace
conda activate pytorch
$target_class = $class_start
$mode = "test"
for (; $target_class -le $class_end; $target_class++){
    # Summary
    $summary = "Target Model [{0}], Target Class [{1}], Batch Size [{2}], Attack Mode [{3}]" -f $model, $target_class, $batch_size, $a_mode
    Write-Output $summary

    # praparing params
    $arg1 = "--mode={0}" -f $mode
    $arg2 = "--attacked_model={0}" -f $model
    $arg3 = "--attack_mode={0}" -f $a_mode
    $arg4 = "--target_class={0}" -f $target_class
    $arg5 = "--batch_size={0}" -f $batch_size
    $arg6 = "--load_pretrained"
    $arg7 = "--pretrained_path=save\{0}-{1}-{2}" -f $model, $a_mode, $target_class
    $arg8 = "--pretrained_step={0}" -f $pretrained_step

    python face_adv.py $arg1, $arg2, $arg3, $arg4, $arg5, $arg6, $arg7, $arg8
}
