#!/usr/bin/make -f
# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
#
# Hidden Valley Mastering Compressor through DPF: VST3 and CLAP (and the AU on macOS, see scripts/build-macos.sh).
# scripts/build.sh and scripts/build-macos.sh run this; by hand: make -C src
# DPF lives in ../third_party/DPF at the commit pinned in scripts/build.sh.

NAME = HiddenValleyMC

FILES_DSP = \
	PluginHVMC.cpp

DPF_PATH ?= ../third_party/DPF
DPF_TARGET_DIR ?= ../build/bin
DPF_BUILD_DIR ?= ../build/obj

# IEEE arithmetic (the fitted constants and the unit tests assume it) and C++17
CXXFLAGS += -std=gnu++17 -fno-fast-math -I.

include $(DPF_PATH)/Makefile.plugins.mk

# the key (sidechain) input needs a second input bus, which DPF's Audio Unit wrapper does not have: scripts/build-macos.sh builds the
# VST3 and CLAP with the key and the AU in a second pass with HVMC_TARGETS=au and -DHVMC_NO_KEY
ifeq ($(MACOS),true)
HVMC_TARGETS ?= vst3 clap au
else
HVMC_TARGETS ?= vst3 clap
endif
TARGETS += $(HVMC_TARGETS)

all: $(TARGETS)
