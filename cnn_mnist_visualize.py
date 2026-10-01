"""本地 MNIST CNN：训练、测试集分类、每个卷积层的特征图。直接点击 VS Code 运行。"""
import os
os.environ['KERAS_BACKEND'] = 'torch'  # 复用本机已验证的 Keras + PyTorch GPU 环境

import csv
import hashlib
import json
import math
import platform
import random
import struct
import sys
from datetime import datetime
from pathlib import Path

import keras
import matplotlib
matplotlib.use('Agg')  # 保存图像，不弹出阻塞训练的窗口
import matplotlib.pyplot as plt
import numpy as np
import torch
from keras import Input, Model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

PROJECT = Path(__file__).resolve().parent
DATA_DIR = PROJECT / 'data' / 'MNIST' / 'raw'
EPOCHS = 5
BATCH_SIZE = 128
SEED = 42
SAMPLE_INDEX = 0  # 修改这里，查看另一张测试图片在各卷积层中的特征


def load_idx(path):
    """读取本地 MNIST IDX 文件；不联网，不下载，不修改原数据。"""
    payload = path.read_bytes()
    zero1, zero2, dtype, ndim = payload[:4]
    if (zero1, zero2, dtype) != (0, 0, 8) or ndim not in (1, 3):
        raise ValueError(f'不是预期的 uint8 MNIST IDX 文件：{path}')
    shape = struct.unpack('>' + 'I' * ndim, payload[4:4 + 4 * ndim])
    values = np.frombuffer(payload, dtype=np.uint8, offset=4 + 4 * ndim)
    if values.size != math.prod(shape):
        raise ValueError(f'IDX 长度与形状不符：{path}')
    return values.reshape(shape), hashlib.sha256(payload).hexdigest()


def cpu_name():
    name = platform.processor() or 'CPU 型号未获取'
    if sys.platform == 'win32':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                name = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    return name


def build_model():
    inputs = Input((28, 28, 1), name='digit')
    # 输出的是经过 ReLU 的卷积特征；特征图中亮处表示该通道响应较强。
    x = Conv2D(16, 3, padding='same', activation='relu', name='conv1')(inputs)
    x = MaxPooling2D(2, name='pool1')(x)
    x = Conv2D(32, 3, padding='same', activation='relu', name='conv2')(x)
    x = MaxPooling2D(2, name='pool2')(x)
    x = Conv2D(64, 3, padding='same', activation='relu', name='conv3')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(64, activation='relu', name='hidden')(x)
    outputs = Dense(10, activation='softmax', name='classifier')(x)
    return Model(inputs, outputs, name='mnist_cnn')


def save_figure(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def visualize(model, images, labels, probabilities, history, folder):
    predicted = probabilities.argmax(axis=1)
    fig, axes = plt.subplots(6, 6, figsize=(10, 11))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'#{i}  true={labels[i]}  pred={predicted[i]}\n'
                     f'p={probabilities[i, predicted[i]]:.1%}', fontsize=9,
                     color='darkgreen' if predicted[i] == labels[i] else 'crimson')
        ax.axis('off')
    fig.suptitle('MNIST test predictions | first 36 samples', fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, folder / 'predictions.png')

    # 额外展示前 36 个错误，避免只挑选正确案例。
    mistakes = np.flatnonzero(predicted != labels)[:36]
    if len(mistakes):
        rows = math.ceil(len(mistakes) / 6)
        fig, axes = plt.subplots(rows, 6, figsize=(10, rows * 1.8), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        for ax, i in zip(axes.flat, mistakes):
            ax.imshow(images[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'#{i} true={labels[i]} pred={predicted[i]}', fontsize=9, color='crimson')
        fig.suptitle('First misclassified test samples', fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save_figure(fig, folder / 'errors.png')

    # 同一张图、训练后的同一个模型，逐层取出所有通道，保证可以相互对照。
    extractor = Model(model.inputs, [model.get_layer(f'conv{i}').output for i in (1, 2, 3)])
    maps = extractor.predict(images[SAMPLE_INDEX:SAMPLE_INDEX + 1], verbose=0)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(images[SAMPLE_INDEX, :, :, 0], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Input #{SAMPLE_INDEX}: true={labels[SAMPLE_INDEX]}, pred={predicted[SAMPLE_INDEX]}')
    ax.axis('off')
    save_figure(fig, folder / 'input_digit.png')
    shapes = {}
    for layer_num, array in enumerate(maps, 1):
        assert np.isfinite(array).all()
        fmap = array[0]
        shapes[f'conv{layer_num}'] = list(fmap.shape)
        channels = fmap.shape[-1]
        cols = 8
        fig, axes = plt.subplots(math.ceil(channels / cols), cols,
                                 figsize=(12, math.ceil(channels / cols) * 1.7), squeeze=False)
        for ch, ax in enumerate(axes.flat):
            ax.axis('off')
            if ch < channels:
                # 每个通道独立缩放，便于看清纹理；跨通道亮度不可直接比较。
                ax.imshow(fmap[:, :, ch], cmap='magma', vmin=0,
                          vmax=max(float(fmap[:, :, ch].max()), 1e-8), interpolation='nearest')
                ax.set_title(f'ch {ch + 1}', fontsize=9)
        fig.suptitle(f'Conv {layer_num} after ReLU | sample #{SAMPLE_INDEX} | '
                     f'{fmap.shape[0]}x{fmap.shape[1]}x{channels}\n'
                     'Each channel scaled independently; black = zero response', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        save_figure(fig, folder / f'conv{layer_num}_features.png')
    np.savez_compressed(folder / 'feature_maps.npz', input=images[SAMPLE_INDEX],
                        **{f'conv{i}': a[0] for i, a in enumerate(maps, 1)})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = np.arange(1, len(history['loss']) + 1)
    for ax, metric in zip(axes, ('loss', 'accuracy')):
        ax.plot(epochs, history[metric], 'o-', label='train')
        ax.plot(epochs, history['val_' + metric], 'o-', label='validation')
        ax.set(xlabel='Epoch', ylabel=metric, xticks=epochs)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, folder / 'training_history.png')
    return shapes


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device_info = dict(python=sys.executable, python_version=platform.python_version(),
                       keras=keras.__version__, backend=keras.backend.backend(),
                       torch=torch.__version__, cuda_build=torch.version.cuda,
                       cpu=cpu_name(), visible_gpus=[torch.cuda.get_device_name(i)
                                                    for i in range(torch.cuda.device_count())],
                       selected_device=str(device))
    print('\n========== 本次训练环境与设备 ==========\n' +
          json.dumps(device_info, ensure_ascii=False, indent=2), flush=True)

    filenames = ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
                 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
    missing = [str(DATA_DIR / name) for name in filenames if not (DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError('请将 DATA_DIR 指向本地 MNIST raw 目录：' + ', '.join(missing))
    loaded = [load_idx(DATA_DIR / name) for name in filenames]
    x_all, y_all, x_test, y_test = [item[0] for item in loaded]
    assert x_all.shape == (60000, 28, 28) and y_all.shape == (60000,)
    assert x_test.shape == (10000, 28, 28) and y_test.shape == (10000,)
    assert np.isin(y_all, np.arange(10)).all() and np.isin(y_test, np.arange(10)).all()
    x_all = x_all.astype('float32')[..., None] / 255.0
    x_test = x_test.astype('float32')[..., None] / 255.0
    y_all, y_test = y_all.astype('int64'), y_test.astype('int64')
    order = np.random.default_rng(SEED).permutation(len(y_all))
    train_idx, val_idx = order[:54000], order[54000:]
    folder = PROJECT / 'cnn_mnist_results' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder / 'split_indices.npz', train=train_idx, validation=val_idx)
    print(f'本地数据：{DATA_DIR}\n训练 54000 / 验证 6000 / 测试 10000', flush=True)

    model = build_model()
    model.to(device)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    initial_kernel = model.trainable_variables[0].value.detach().clone()
    history = model.fit(x_all[train_idx], y_all[train_idx],
                        validation_data=(x_all[val_idx], y_all[val_idx]),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2).history
    assert not torch.equal(initial_kernel, model.trainable_variables[0].value.detach())
    probabilities = model.predict(x_test, batch_size=256, verbose=0)
    assert probabilities.shape == (10000, 10) and np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    predictions = probabilities.argmax(axis=1)
    accuracy = float((predictions == y_test).mean())
    test_loss = float(-np.log(np.clip(probabilities[np.arange(10000), y_test], 1e-7, 1)).mean())
    print(f'\n独立测试集准确率：{accuracy:.2%}（共 10000 张）', flush=True)
    print('前 100 张测试图片分类结果（编号从 0 开始；全部结果保存到 predictions.csv）：')
    for i in range(100):
        print(f'图片 {i:04d}：真实={y_test[i]}，预测={predictions[i]}，'
              f'预测概率={probabilities[i, predictions[i]]:.2%}')
    with (folder / 'predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['test_index', 'true_label', 'predicted_label', 'correct', 'confidence'] +
                        [f'p_{i}' for i in range(10)])
        for i in range(len(y_test)):
            writer.writerow([i, int(y_test[i]), int(predictions[i]), bool(y_test[i] == predictions[i]),
                             float(probabilities[i, predictions[i]])] + probabilities[i].tolist())
    shapes = visualize(model, x_test, y_test, probabilities, history, folder)
    model.save(folder / 'mnist_cnn.keras')
    device_info['actual_parameter_devices'] = sorted({str(p.value.device) for p in model.trainable_variables})
    summary = dict(version='v001', finished_at=datetime.now().astimezone().isoformat(),
                   device=device_info, data_dir=str(DATA_DIR),
                   data_sha256={name: value[1] for name, value in zip(filenames, loaded)},
                   seed=SEED, epochs=EPOCHS, batch_size=BATCH_SIZE,
                   train_size=54000, validation_size=6000, test_size=10000,
                   history={k: [float(v) for v in values] for k, values in history.items()},
                   test_accuracy=accuracy, test_cross_entropy=test_loss,
                   correct_count=int((predictions == y_test).sum()),
                   parameters=model.count_params(), feature_sample_index=SAMPLE_INDEX,
                   feature_sample_true=int(y_test[SAMPLE_INDEX]),
                   feature_sample_prediction=int(predictions[SAMPLE_INDEX]), feature_shapes=shapes,
                   checks='IDX format, labels, parameter update, 10000 probability rows, finite feature maps passed')
    (folder / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    sections = ''.join(f'<h2>卷积层 {i}：全部 {shapes[f"conv{i}"][-1]} 个通道</h2>'
                       f'<img src="conv{i}_features.png" alt="卷积层 {i} 特征图">' for i in (1, 2, 3))
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MNIST CNN 训练结果</title>
    <style>body{{max-width:1100px;margin:36px auto;padding:0 24px;font:16px/1.7 sans-serif;color:#162033}}
    img{{max-width:100%;height:auto;border:1px solid #dde3ee;border-radius:10px}}h2{{margin-top:36px}}</style>
    <h1>MNIST CNN 训练与卷积层特征</h1>
    <p>独立测试集：{accuracy:.2%}，正确 {summary['correct_count']} / 10000。
    训练 54000 张，验证 6000 张，训练 {EPOCHS} 轮。</p>
    <p>实际参数设备：{', '.join(device_info['actual_parameter_devices'])}；
    物理训练设备：{torch.cuda.get_device_name(device) if device.type == 'cuda' else device_info['cpu']}。</p>
    <p><a href="predictions.csv">全部 10000 张分类结果</a> · <a href="summary.json">训练记录</a> ·
    <a href="mnist_cnn.keras">训练后的模型</a> · <a href="feature_maps.npz">原始特征数值</a></p>
    <h2>前 36 张测试图片的分类结果</h2><img src="predictions.png" alt="分类结果">
    <h2>各层观察的同一张输入图片</h2><img src="input_digit.png" alt="原始数字" style="max-width:360px">
    <p>下图是训练后各卷积层经过 ReLU 的激活特征图，不是各层独立作出的分类。
    所有通道均显示；每张小图单独调整显示亮度以看清结构，不能直接比较不同通道的颜色强弱。
    黑色表示该位置响应为零。后面的层尺寸更小、通道更多，不保证仍呈现完整数字轮廓。</p>
    {sections}<h2>训练与验证曲线</h2><img src="training_history.png" alt="训练曲线">
    {'<h2>前 36 个错误预测</h2><img src="errors.png" alt="错误案例">' if (folder / 'errors.png').exists() else ''}
    </html>'''
    (folder / 'report.html').write_text(html, encoding='utf-8')
    print('\n========== 实际训练设备 ==========')
    print(json.dumps(device_info, ensure_ascii=False, indent=2))
    print(f'\n结果目录：{folder}\n双击 report.html 查看分类结果与每个卷积层的特征图。', flush=True)


if __name__ == '__main__':
    main()
