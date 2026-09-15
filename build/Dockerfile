# vdb_tuning build/run environment.
# Ubuntu 24.04 base. The ubuntu:24.04 tag is a multi-arch manifest, so this
# image builds correctly on both x86_64 and arm64 (Graviton) hosts.
FROM ubuntu:24.04

# Avoid interactive tzdata/apt prompts during build.
ENV DEBIAN_FRONTEND=noninteractive

# All build + runtime dependencies for PostgreSQL 18, pgvector, TPC-H dbgen,
# and the Python experiment tooling. No architecture-specific packages here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        make \
        libreadline-dev \
        zlib1g-dev \
        libicu-dev \
        pkg-config \
        flex \
        bison \
        libxml2-dev \
        libssl-dev \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        numactl \
        git \
        wget \
        curl \
        ca-certificates \
        sudo \
        procps \
        vim-tiny \
        locales \
        pigz \
    && rm -rf /var/lib/apt/lists/*

# The repo's postgresql.conf sets lc_messages/lc_monetary/etc. to en_US.UTF-8,
# so that locale must exist in the image (Ubuntu ships only C.UTF-8 by default).
RUN sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# Create the postgres OS user with passwordless sudo (setup scripts use sudo/su).
RUN id -u postgres >/dev/null 2>&1 || useradd -m -s /bin/bash postgres \
    && echo "postgres ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/postgres \
    && chmod 0440 /etc/sudoers.d/postgres

# Python virtualenv used by the experiment scripts, created at a fixed path so
# it does not collide with the bind-mounted repo's own .venv.
ENV VENV=/opt/vdb-venv
RUN python3 -m venv "${VENV}" \
    && "${VENV}/bin/pip" install --no-cache-dir --upgrade pip \
    && "${VENV}/bin/pip" install --no-cache-dir \
        psycopg2-binary \
        "psycopg[binary]" \
        pgvector \
        h5py \
        numpy \
        psutil \
        matplotlib \
        pandas \
        pyyaml

# Make Postgres binaries and the venv available on PATH; PGDATA per config.
ENV PATH=/usr/local/pgsql/bin:/opt/vdb-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PGDATA=/mnt/nvme/pgdata
# The scripts read machine paths from this config (container-path variant).
ENV VDB_CONFIG=/vdb_tuning/build/config.container.yaml

EXPOSE 5432

WORKDIR /vdb_tuning
CMD ["/bin/bash"]
