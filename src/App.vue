<template>
  <div id="app">
    <div>
      <h1> 勝手にぬりえ </h1>
    </div>
    <NurieView modelPath="/jsmodel/model.json" />
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator'
import NurieView from './components/NurieView.vue'

@Component({
  components: {
    NurieView
  },
})
export default class App extends Vue {}
</script>

<style lang="scss">
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
