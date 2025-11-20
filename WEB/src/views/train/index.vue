<template>
  <div class="train-wrapper">
    <div class="train-tab">
      <Tabs
        :animated="{ inkBar: true, tabPane: true }"
        :activeKey="state.activeKey"
        :tabBarGutter="60"
        @tabClick="handleTabClick"
      >
        <TabPane key="1" tab="模型管理">
          <ModelList></ModelList>
        </TabPane>
        <TabPane key="2" tab="模型推理">
          <AiModelTool></AiModelTool>
        </TabPane>
        <TabPane key="3" tab="模型导出">
          <ModelExport></ModelExport>
        </TabPane>
      </Tabs>
    </div>
  </div>
</template>

<script lang="ts" setup name="TrainService">
import {reactive} from 'vue';
import { TabPane, Tabs } from "ant-design-vue";
import ModelList from "@/views/train/components/ModelList/index.vue";
import AiModelTool from "@/views/train/components/AiModelTool/index.vue";
import ModelExport from "@/views/train/components/ModelExport/index.vue";

defineOptions({name: 'TRAIN'})

const state = reactive({
  activeKey: '1'
});

const handleTabClick = (activeKey: string) => {
  state.activeKey = activeKey;
};
</script>

<style lang="less" scoped>
.train-wrapper {
  :deep(.ant-tabs-nav) {
    padding: 5px 0 0 25px;
  }

  .train-tab {
    padding: 16px 19px 0 15px;

    .ant-tabs {
      background-color: #FFFFFF;

      :deep(.ant-tabs-nav) {
        padding: 5px 0 0 25px;
      }
    }
  }
}
</style>
