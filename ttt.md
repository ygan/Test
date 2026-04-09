你这次这个结果其实说明了关键点：

你用的这个

```bash id="rd5su4"
/project/p311/torch_wheels/test/lib/python3.10/site-packages/certifi/cacert.pem
```

是 **certifi 自带的公共 CA 列表**，它**不会**包含学校/TRE 自己的代理证书或私有根证书。
所以如果你们出口 HTTPS 被学校/TRE 做了中间代理，`--cert certifi/cacert.pem` 基本一定还是报：

```text
CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate
```

现在别再试 certifi 那个 pem 了。直接用下面这套。

---

## 方案 A：先用 3.10 下载 `truststore`，再装进 3.11

这个最值得先试。

### 在能工作的 3.10 环境里跑

```bash id="7mjaon"
mkdir -p ~/wheelhouse
python -m pip download -d ~/wheelhouse truststore packaging
ls -l ~/wheelhouse
```

### 然后切到 3.11 环境里跑

```bash id="na8p0r"
python -m pip install --no-index --find-links ~/wheelhouse truststore
```

装好以后测试：

```bash id="17evda"
python -m pip download packaging --no-deps --use-feature=truststore -v
```

如果这条成功，再装你要的包：

```bash id="4ys6e5"
python -m pip install openai --use-feature=truststore -v
```

---

## 方案 B：直接试系统 CA bundle，不用 certifi

先在 3.11 里看系统默认 CA 路径：

```bash id="abn5o5"
python - <<'PY'
import ssl
print(ssl.get_default_verify_paths())
PY
```

然后直接把 Linux 上常见的系统 CA 文件全试一遍：

```bash id="hycgrs"
for f in \
/etc/ssl/certs/ca-certificates.crt \
/etc/pki/tls/certs/ca-bundle.crt \
/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem \
/etc/ssl/cert.pem
do
  if [ -f "$f" ]; then
    echo "===== TRY $f ====="
    python -m pip download packaging --no-deps --cert "$f" -v && break
  fi
done
```

如果其中某个路径成功了，再正式安装：

```bash id="c6c6l7"
python -m pip install openai --cert /上一步成功的那个文件路径 -v
```

然后固化到 pip 配置：

```bash id="0i4zeb"
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf <<'EOF'
[global]
cert = /上一步成功的那个文件路径
EOF
cat ~/.config/pip/pip.conf
```

---

## 方案 C：从系统里直接搜学校/TRE 的证书

如果 A 和 B 都不行，就把机器上的证书文件先找出来：

```bash id="mrtjmb"
find /etc "$HOME" -type f \( -name '*.pem' -o -name '*.crt' \) 2>/dev/null | \
grep -Ei 'ca|bundle|trust|cert|proxy|mitm|tre|university|school'
```

然后拿找到的候选证书逐个测试：

```bash id="n8wmfo"
python -m pip download packaging --no-deps --cert /找到的某个证书文件 -v
```

哪个能通，就把哪个写进 `~/.config/pip/pip.conf`。

---

## 我建议你现在就按这个顺序跑

先跑这 3 句：

```bash id="gj56j5"
mkdir -p ~/wheelhouse
python -m pip download -d ~/wheelhouse truststore packaging
```

切到 3.11 后：

```bash id="tt7f75"
python -m pip install --no-index --find-links ~/wheelhouse truststore
python -m pip download packaging --no-deps --use-feature=truststore -v
```

然后把最后一条命令的结果告诉我：
是成功下载了，还是还报 `CERTIFICATE_VERIFY_FAILED`。
