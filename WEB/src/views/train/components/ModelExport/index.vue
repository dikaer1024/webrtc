<template>
  <div class="model-export-container">
    <a-row :gutter="24" class="export-layout">
      <!-- 左侧配置面板 -->
      <a-col :span="8" class="left-panel">
        <a-card 
          title="导出配置" 
          :bordered="false" 
          class="config-card"
          :head-style="{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', border: 'none' }"
          :body-style="{ padding: '32px' }"
        >
          <template #title>
            <div class="card-title">
              <ExportOutlined class="title-icon" />
              <span>模型导出配置</span>
            </div>
          </template>

          <!-- 模型选择 -->
          <div class="config-section">
            <div class="section-header">
              <span class="section-title">选择模型</span>
              <span class="section-badge" v-if="models.length > 0">{{ models.length }} 个可用</span>
            </div>
            
            <a-alert
              v-if="models.length === 0"
              message="暂无可用模型"
              description="当前没有可导出的 pt 模型，只有 pt 格式的模型才能导出为 onnx 或 openvino 格式"
              type="info"
              show-icon
              :closable="false"
              class="alert-box"
            />
            
            <a-select
              v-model:value="selectedModelId"
              placeholder="请选择要导出的 pt 模型"
              style="width: 100%"
              size="large"
              show-search
              :filter-option="filterModelOption"
              @change="handleModelChange"
              :disabled="models.length === 0"
              class="model-select"
            >
              <a-select-option
                v-for="model in models"
                :key="model.id"
                :value="model.id"
              >
                <div class="model-option">
                  <span class="model-name">{{ model.name }}</span>
                  <span class="model-version">v{{ model.version }}</span>
                </div>
              </a-select-option>
            </a-select>
            
            <div class="select-tip" v-if="models.length > 0">
              <InfoCircleOutlined />
              <span>仅显示 pt 格式的模型，onnx 和 openvino 模型不支持导出</span>
            </div>
            
            <!-- 模型信息展示 -->
            <div v-if="selectedModel" class="model-info-card">
              <div class="info-header">
                <FileTextOutlined />
                <span>模型信息</span>
              </div>
              <div class="info-content">
                <div class="info-item">
                  <span class="info-label">模型名称</span>
                  <span class="info-value">{{ selectedModel.name }}</span>
                </div>
                <div class="info-item">
                  <span class="info-label">模型版本</span>
                  <span class="info-value">v{{ selectedModel.version }}</span>
                </div>
                <div class="info-item" v-if="selectedModel.description">
                  <span class="info-label">描述</span>
                  <span class="info-value">{{ selectedModel.description }}</span>
                </div>
              </div>
            </div>
          </div>

          <!-- 导出格式选择 -->
          <div class="config-section" v-if="selectedModelId">
            <div class="section-header">
              <span class="section-title">导出格式</span>
            </div>
            <a-radio-group 
              v-model:value="exportForm.format" 
              class="format-group"
              size="large"
            >
              <a-radio-button value="onnx" class="format-option">
                <div class="format-content">
                  <span class="format-icon">ONNX</span>
                  <span class="format-desc">通用格式</span>
                </div>
              </a-radio-button>
              <a-radio-button value="openvino" class="format-option">
                <div class="format-content">
                  <span class="format-icon">OpenVINO</span>
                  <span class="format-desc">Intel优化</span>
                </div>
              </a-radio-button>
            </a-radio-group>
          </div>

          <!-- 导出参数配置 -->
          <div class="config-section" v-if="selectedModelId && exportForm.format">
            <div class="section-header">
              <span class="section-title">导出参数</span>
            </div>
            <a-form
              :model="exportForm"
              layout="vertical"
              class="export-form"
            >
              <a-form-item label="图像尺寸" class="form-item-custom">
                <a-input-number
                  v-model:value="exportForm.img_size"
                  :min="128"
                  :max="1024"
                  :step="32"
                  style="width: 100%"
                  size="large"
                  placeholder="推荐值：640"
                />
                <div class="form-tip">
                  <BulbOutlined />
                  <span>推荐值：640，支持 128-1024 之间的值</span>
                </div>
              </a-form-item>

              <a-form-item label="OPSet版本" v-if="exportForm.format === 'onnx'" class="form-item-custom">
                <a-input-number
                  v-model:value="exportForm.opset"
                  :min="10"
                  :max="15"
                  style="width: 100%"
                  size="large"
                  placeholder="推荐值：12"
                />
                <div class="form-tip">
                  <BulbOutlined />
                  <span>推荐值：12，支持 10-15 之间的值</span>
                </div>
              </a-form-item>
            </a-form>
          </div>

          <!-- 导出按钮 -->
          <div class="config-section action-section" v-if="selectedModelId">
            <a-button
              type="primary"
              :loading="exportLoading"
              :disabled="!exportForm.format"
              @click="handleExport"
              block
              size="large"
              class="export-button"
            >
              <template #icon>
                <DownloadOutlined />
              </template>
              {{ exportLoading ? '导出中...' : '开始导出' }}
            </a-button>
            <a-button
              block
              size="large"
              @click="resetForm"
              class="reset-button"
            >
              <template #icon>
                <ReloadOutlined />
              </template>
              重置配置
            </a-button>
          </div>
        </a-card>
      </a-col>

      <!-- 右侧导出历史记录 -->
      <a-col :span="16" class="right-panel">
        <a-card 
          title="导出历史记录" 
          :bordered="false" 
          class="history-card"
          :body-style="{ padding: '24px' }"
        >
          <template #title>
            <div class="card-title">
              <HistoryOutlined class="title-icon" />
              <span>导出历史记录</span>
            </div>
          </template>
          <template #extra>
            <a-space>
              <a-button 
                type="text" 
                @click="reloadExportList" 
                :loading="tableLoading"
                class="refresh-btn"
              >
                <template #icon>
                  <ReloadOutlined :spin="tableLoading" />
                </template>
                刷新
              </a-button>
            </a-space>
          </template>
          
          <BasicTable
            @register="registerTable"
            :bordered="true"
            :row-selection="false"
            class="export-table"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.dataIndex === 'format'">
                <a-tag :color="formatColors[record.format]" class="format-tag">
                  {{ formatLabels[record.format] || record.format.toUpperCase() }}
                </a-tag>
              </template>
              <template v-else-if="column.dataIndex === 'status'">
                <a-badge 
                  :status="getStatusBadgeStatus(record.status)" 
                  :text="statusLabels[record.status] || record.status"
                  class="status-badge"
                />
              </template>
              <template v-else-if="column.dataIndex === 'created_at'">
                <div class="time-cell">
                  <ClockCircleOutlined />
                  <span>{{ formatDate(record.created_at) }}</span>
                </div>
              </template>
              <template v-else-if="column.dataIndex === 'action'">
                <a-space>
                  <a-button
                    type="link"
                    size="small"
                    :disabled="record.status !== 'COMPLETED'"
                    @click="handleDownload(record)"
                    class="action-btn"
                  >
                    <template #icon>
                      <DownloadOutlined />
                    </template>
                    下载
                  </a-button>
                  <a-popconfirm
                    title="确定删除此导出记录吗？"
                    ok-text="确认"
                    cancel-text="取消"
                    @confirm="handleDelete(record)"
                  >
                    <a-button
                      type="link"
                      size="small"
                      danger
                      class="action-btn"
                    >
                      <template #icon>
                        <DeleteOutlined />
                      </template>
                      删除
                    </a-button>
                  </a-popconfirm>
                </a-space>
              </template>
            </template>
          </BasicTable>
        </a-card>
      </a-col>
    </a-row>
  </div>
</template>

<script lang="ts" setup>
import { ref, reactive, computed, onMounted, onUnmounted } from 'vue';
import { BasicTable, useTable } from '@/components/Table';
import { message } from 'ant-design-vue';
import { 
  DownloadOutlined, 
  ReloadOutlined, 
  InfoCircleOutlined,
  ExportOutlined,
  FileTextOutlined,
  HistoryOutlined,
  BulbOutlined,
  ClockCircleOutlined,
  DeleteOutlined
} from '@ant-design/icons-vue';
import {
  getModelPage,
  exportModel,
  getExportModelList,
  deleteExportedModel,
  downloadExportedModel,
  getExportStatus,
} from '@/api/device/model';
import dayjs from 'dayjs';

// 格式标签映射
const formatLabels = {
  onnx: 'ONNX',
  openvino: 'OpenVINO',
};

// 格式颜色映射
const formatColors = {
  onnx: 'green',
  openvino: 'cyan',
};

// 状态标签映射
const statusLabels = {
  PENDING: '等待中',
  PROCESSING: '处理中',
  COMPLETED: '已完成',
  FAILED: '失败',
};

// 状态徽章映射
const getStatusBadgeStatus = (status: string) => {
  const statusMap: Record<string, 'default' | 'processing' | 'success' | 'error'> = {
    PENDING: 'default',
    PROCESSING: 'processing',
    COMPLETED: 'success',
    FAILED: 'error',
  };
  return statusMap[status] || 'default';
};

// 数据状态
const models = ref<any[]>([]);
const selectedModelId = ref<number | null>(null);
const exportLoading = ref(false);
const tableLoading = ref(false);
const pollingIntervals = ref<Map<number | string, NodeJS.Timeout>>(new Map());

// 导出表单
const exportForm = reactive({
  format: 'onnx',
  img_size: 640,
  opset: 12,
});

// 选中的模型
const selectedModel = computed(() => {
  return models.value.find((m: any) => m.id === selectedModelId.value);
});

// 表格配置
const [registerTable, { reload: reloadExportList }] = useTable({
  title: '',
  api: async (params) => {
    tableLoading.value = true;
    try {
      const { page, pageSize } = params;
      const res = await getExportModelList({
        model_id: selectedModelId.value || undefined,
        page: page || 1,
        per_page: pageSize || 10,
      });
      
      if (res.success && res.data) {
        return {
          list: res.data.items || [],
          total: res.data.total || 0,
        };
      }
      
      return {
        list: res.data?.items || res.data?.list || [],
        total: res.data?.total || 0,
      };
    } catch (error: any) {
      console.error('获取导出列表失败:', error);
      message.error('获取导出列表失败');
      return {
        list: [],
        total: 0,
      };
    } finally {
      tableLoading.value = false;
    }
  },
  columns: [
    {
      title: 'ID',
      dataIndex: 'id',
      width: 80,
      align: 'center',
    },
    {
      title: '模型ID',
      dataIndex: 'model_id',
      width: 100,
      align: 'center',
    },
    {
      title: '导出格式',
      dataIndex: 'format',
      width: 120,
      align: 'center',
    },
    {
      title: '状态',
      dataIndex: 'status',
      width: 120,
      align: 'center',
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      width: 180,
      align: 'center',
    },
    {
      title: '操作',
      dataIndex: 'action',
      width: 150,
      align: 'center',
      fixed: 'right',
    },
  ],
  useSearchForm: false,
  rowKey: 'id',
  pagination: { pageSize: 10 },
});

// 判断是否为 pt 模型（可导出的模型）
const isPtModel = (model: any): boolean => {
  if (!model.model_path) {
    return false;
  }
  
  const modelPath = model.model_path.toLowerCase();
  return !modelPath.endsWith('.onnx') && (modelPath.endsWith('.pt') || !model.onnx_model_path);
};

// 加载模型列表（只加载 pt 格式的模型）
const loadModels = async () => {
  try {
    const response = await getModelPage({ page: 1, size: 100 });
    if (response.code === 0) {
      const allModels = response.data || [];
      models.value = allModels.filter(isPtModel);
      
      if (models.value.length === 0) {
        message.info('当前没有可导出的 pt 模型');
      }
    }
  } catch (error: any) {
    console.error('加载模型列表失败:', error);
    message.error('加载模型列表失败');
  }
};

// 模型选择过滤
const filterModelOption = (input: string, option: any) => {
  return option.children.toLowerCase().indexOf(input.toLowerCase()) >= 0;
};

// 模型变更处理
const handleModelChange = () => {
  resetForm();
  reloadExportList();
};

// 重置表单
const resetForm = () => {
  exportForm.format = 'onnx';
  exportForm.img_size = 640;
  exportForm.opset = 12;
};

// 执行导出
const handleExport = async () => {
  if (!selectedModelId.value) {
    message.warning('请先选择模型');
    return;
  }

  if (!exportForm.format) {
    message.warning('请选择导出格式');
    return;
  }

  exportLoading.value = true;
  try {
    const params = {
      img_size: exportForm.img_size,
      opset: exportForm.opset,
    };

    const res = await exportModel(selectedModelId.value, exportForm.format, params);
    
    if (res.success || res.code === 0 || res.data) {
      message.success('导出任务已提交，请稍后刷新查看');
      
      // 刷新列表
      reloadExportList();
      
      // 开始轮询状态
      const taskId = res.task_id || res.data?.task_id;
      const exportId = res.export_id || res.data?.export_id;
      if (taskId) {
        startPollingByTaskId(taskId);
      } else if (exportId) {
        startPolling(exportId);
      }
    } else {
      throw new Error(res.message || '导出失败');
    }
  } catch (error: any) {
    console.error('导出失败:', error);
    message.error(error.message || '导出失败，请重试');
  } finally {
    exportLoading.value = false;
  }
};

// 状态轮询（通过export_id）
const startPolling = (exportId: number) => {
  if (pollingIntervals.value.has(exportId)) {
    clearInterval(pollingIntervals.value.get(exportId)!);
  }

  const interval = setInterval(async () => {
    try {
      const statusRes = await getExportStatus(exportId);
      
      if (statusRes.status === 'COMPLETED' || statusRes.status === 'FAILED') {
        clearInterval(interval);
        pollingIntervals.value.delete(exportId);
        
        if (statusRes.status === 'FAILED') {
          message.error(`导出失败: ${statusRes.error || '未知错误'}`);
        } else {
          message.success('导出完成');
        }
        
        reloadExportList();
      }
    } catch (error) {
      console.error('状态检查失败', error);
    }
  }, 5000);

  pollingIntervals.value.set(exportId, interval);
};

// 状态轮询（通过task_id）
const startPollingByTaskId = (taskId: string) => {
  if (pollingIntervals.value.has(taskId)) {
    clearInterval(pollingIntervals.value.get(taskId)!);
  }

  const interval = setInterval(async () => {
    try {
      const statusRes = await getExportStatus(taskId);
      
      if (statusRes.status === 'COMPLETED' || statusRes.status === 'FAILED') {
        clearInterval(interval);
        pollingIntervals.value.delete(taskId);
        
        if (statusRes.status === 'FAILED') {
          message.error(`导出失败: ${statusRes.error || '未知错误'}`);
        } else {
          message.success('导出完成');
        }
        
        reloadExportList();
      }
    } catch (error) {
      console.error('状态检查失败', error);
    }
  }, 5000);

  pollingIntervals.value.set(taskId, interval);
};

// 下载导出文件
const handleDownload = async (record: any) => {
  if (record.status !== 'COMPLETED') {
    message.warning('导出任务尚未完成，请稍后再试');
    return;
  }

  try {
    const response = await downloadExportedModel(record.id);
    
    // 处理blob响应
    let blob: Blob;
    if (response instanceof Blob) {
      blob = response;
    } else if (response.data instanceof Blob) {
      blob = response.data;
    } else {
      throw new Error('无效的响应格式');
    }
    
    // 创建下载链接
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    
    // 从响应头或记录中获取文件名
    const fileName = `${record.model_id || 'model'}_${record.format}.${getFileExtension(record.format)}`;
    link.download = fileName;
    
    document.body.appendChild(link);
    link.click();
    
    // 清理资源
    setTimeout(() => {
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }, 100);

    message.success('文件下载成功');
  } catch (error: any) {
    console.error('下载失败:', error);
    message.error(error.message || '下载失败，请重试');
  }
};

// 删除导出记录
const handleDelete = async (record: any) => {
  try {
    await deleteExportedModel(record.id);
    message.success('导出记录已删除');
    reloadExportList();
  } catch (error: any) {
    console.error('删除失败:', error);
    message.error(error.message || '删除失败，请重试');
  }
};

// 获取文件扩展名
const getFileExtension = (format: string) => {
  const extensions: Record<string, string> = {
    onnx: 'onnx',
    openvino: 'zip', // OpenVINO导出为目录，通常打包为zip
  };
  return extensions[format] || 'bin';
};

// 格式化日期
const formatDate = (dateString: string) => {
  if (!dateString) return '--';
  return dayjs(dateString).format('YYYY-MM-DD HH:mm:ss');
};

// 初始化
onMounted(() => {
  loadModels();
  reloadExportList();
});

// 组件卸载时清理轮询
onUnmounted(() => {
  pollingIntervals.value.forEach((interval) => {
    clearInterval(interval);
  });
  pollingIntervals.value.clear();
});
</script>

<style lang="less" scoped>
.model-export-container {
  padding: 24px;
  background: #f0f2f5;
  min-height: calc(100vh - 200px);

  .export-layout {
    height: 100%;
  }

  // 左侧配置面板
  .left-panel {
    .config-card {
      height: 100%;
      border-radius: 8px;
      box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
      overflow: hidden;

      :deep(.ant-card-head) {
        border-bottom: none;
        padding: 0;

        .ant-card-head-title {
          padding: 20px 24px;
        }
      }

      .card-title {
        display: flex;
        align-items: center;
        gap: 12px;
        color: #fff;
        font-size: 18px;
        font-weight: 600;

        .title-icon {
          font-size: 20px;
        }
      }

      .config-section {
        margin-bottom: 32px;

        &:last-child {
          margin-bottom: 0;
        }

        .section-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 16px;

          .section-title {
            font-size: 15px;
            font-weight: 600;
            color: #1f2c3d;
            position: relative;
            padding-left: 12px;

            &::before {
              content: '';
              position: absolute;
              left: 0;
              top: 50%;
              transform: translateY(-50%);
              width: 3px;
              height: 16px;
              background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
              border-radius: 2px;
            }
          }

          .section-badge {
            padding: 2px 8px;
            background: #f0f0f0;
            border-radius: 12px;
            font-size: 12px;
            color: #666;
          }
        }

        .alert-box {
          margin-bottom: 16px;
          border-radius: 6px;
        }

        .model-select {
          :deep(.ant-select-selector) {
            border-radius: 6px;
            border: 1px solid #d9d9d9;
            transition: all 0.3s;

            &:hover {
              border-color: #667eea;
            }
          }

          :deep(.ant-select-focused .ant-select-selector) {
            border-color: #667eea;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
          }
        }

        .model-option {
          display: flex;
          justify-content: space-between;
          align-items: center;

          .model-name {
            font-weight: 500;
            color: #1f2c3d;
          }

          .model-version {
            font-size: 12px;
            color: #8c8c8c;
            background: #f5f5f5;
            padding: 2px 8px;
            border-radius: 4px;
          }
        }

        .select-tip {
          margin-top: 12px;
          padding: 10px 14px;
          background: #e6f7ff;
          border: 1px solid #91d5ff;
          border-radius: 6px;
          font-size: 13px;
          color: #1890ff;
          display: flex;
          align-items: center;
          gap: 8px;
          line-height: 1.5;

          :deep(.anticon) {
            font-size: 14px;
          }
        }

        .model-info-card {
          margin-top: 16px;
          padding: 16px;
          background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
          border-radius: 8px;
          border: 1px solid #e8e8e8;

          .info-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
            font-weight: 600;
            color: #1f2c3d;
            font-size: 14px;

            :deep(.anticon) {
              color: #667eea;
            }
          }

          .info-content {
            .info-item {
              display: flex;
              margin-bottom: 10px;
              font-size: 14px;
              line-height: 1.6;

              &:last-child {
                margin-bottom: 0;
              }

              .info-label {
                color: #666;
                min-width: 80px;
                font-weight: 500;
              }

              .info-value {
                color: #1f2c3d;
                flex: 1;
                word-break: break-word;
              }
            }
          }
        }

        .format-group {
          width: 100%;
          display: flex;
          gap: 12px;

          :deep(.ant-radio-button-wrapper) {
            flex: 1;
            height: 70px;
            border-radius: 8px;
            border: 2px solid #e8e8e8;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #fff;

            &:first-child {
              border-radius: 8px;
            }

            &:last-child {
              border-radius: 8px;
            }

            &:hover {
              border-color: #667eea;
              transform: translateY(-2px);
              box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
            }

            &.ant-radio-button-wrapper-checked {
              background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
              border-color: #667eea;
              color: #fff;
              box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);

              .format-content {
                .format-icon {
                  color: #fff;
                }

                .format-desc {
                  color: rgba(255, 255, 255, 0.9);
                }
              }
            }

            .format-content {
              display: flex;
              flex-direction: column;
              align-items: center;
              gap: 4px;

              .format-icon {
                font-size: 16px;
                font-weight: 600;
                color: #1f2c3d;
              }

              .format-desc {
                font-size: 12px;
                color: #8c8c8c;
              }
            }
          }
        }

        .export-form {
          :deep(.ant-form-item) {
            margin-bottom: 24px;

            .ant-form-item-label {
              padding-bottom: 8px;

              label {
                font-weight: 500;
                color: #1f2c3d;
                font-size: 14px;
              }
            }
          }

          .form-item-custom {
            :deep(.ant-input-number) {
              border-radius: 6px;
            }
          }

          .form-tip {
            margin-top: 8px;
            color: #8c8c8c;
            font-size: 12px;
            line-height: 1.5;
            display: flex;
            align-items: center;
            gap: 6px;

            :deep(.anticon) {
              color: #faad14;
            }
          }
        }

        &.action-section {
          margin-top: 32px;
          padding-top: 24px;
          border-top: 1px solid #f0f0f0;
        }

        .export-button {
          height: 48px;
          font-size: 16px;
          font-weight: 500;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          border: none;
          border-radius: 6px;
          box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
          transition: all 0.3s;

          &:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
          }

          &:active:not(:disabled) {
            transform: translateY(0);
          }

          &:disabled {
            opacity: 0.6;
          }
        }

        .reset-button {
          margin-top: 12px;
          height: 44px;
          border-radius: 6px;
          border: 1px solid #d9d9d9;
          transition: all 0.3s;

          &:hover {
            border-color: #667eea;
            color: #667eea;
          }
        }
      }
    }
  }

  // 右侧历史记录面板
  .right-panel {
    .history-card {
      height: 100%;
      border-radius: 8px;
      box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);

      :deep(.ant-card-head) {
        border-bottom: 1px solid #f0f0f0;
        padding: 20px 24px;

        .ant-card-head-title {
          padding: 0;
        }

        .ant-card-extra {
          padding: 0;
        }
      }

      .card-title {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 18px;
        font-weight: 600;
        color: #1f2c3d;

        .title-icon {
          font-size: 20px;
          color: #667eea;
        }
      }

      .refresh-btn {
        color: #667eea;

        &:hover {
          color: #764ba2;
        }
      }

      .export-table {
        :deep(.ant-table) {
          .ant-table-thead > tr > th {
            background: #fafafa;
            font-weight: 600;
            color: #1f2c3d;
            border-bottom: 2px solid #f0f0f0;
          }

          .ant-table-tbody > tr {
            transition: all 0.3s;

            &:hover {
              background: #f8f9fa;
            }
          }
        }

        .format-tag {
          font-weight: 500;
          padding: 4px 12px;
          border-radius: 4px;
        }

        .status-badge {
          :deep(.ant-badge-status-text) {
            font-size: 13px;
            color: #1f2c3d;
          }
        }

        .time-cell {
          display: flex;
          align-items: center;
          gap: 6px;
          color: #666;
          font-size: 13px;

          :deep(.anticon) {
            color: #8c8c8c;
          }
        }

        .action-btn {
          padding: 0;
          height: auto;
          font-size: 13px;

          &:hover {
            color: #667eea;
          }
        }
      }
    }
  }

  // 响应式布局
  @media (max-width: 1400px) {
    .left-panel {
      margin-bottom: 24px;
    }
  }
}
</style>
