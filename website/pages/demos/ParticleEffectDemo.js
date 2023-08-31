/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the scripts directory.
 *
 * @format
 */

const CWD = process.cwd();

const React = require('react');
const Demo = require(`${CWD}/core/Demo.js`);

class DemoPage extends React.Component {
  render() {
    const { config: siteConfig } = this.props;
    const { baseUrl } = siteConfig;
    return <Demo baseUrl={baseUrl} demoID="ParticleEffectDemo" hasWinDemo="False"
      hasMacDemo="False" />;
  }
}

module.exports = DemoPage;
